import atexit
import gc
import math
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import wandb

from src import train_utils
from src import viz
from src.dataloader import train_dataloader, val_dataloader
from src.models import PJPE, weight_init, Critic
from src.trainer import training_epoch, validation_epoch
from src.callbacks import CallbackList, ModelCheckpoint, Logging, BetaScheduler, Analyze


def main():

    # Experiment Configuration, Config, is distributed to all the other modules
    parser = training_specific_args()
    config = parser.parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # log intervals
    config.eval_interval = 1  # interval to get MPJPE of 3d decoder
    config.manifold_interval = 1  # interval to visualize encoding in manifold

    # GPU setup
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.device = device  # Adding device to config, not already in argparse
    config.num_workers = 4 if use_cuda else 4  # for dataloader

    # wandb for experiment monitoring
    os.environ['WANDB_NOTES'] = 'None'

    # ignore when debugging on cpu
    if not use_cuda:
        os.environ['WANDB_MODE'] = 'dryrun'  # Doesnt auto sync to project
        os.environ['WANDB_TAGS'] = 'CPU'
        wandb.init(anonymous='allow', project="to_delete", config=config)
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
        wandb.init(anonymous='allow', project="hpe3d", config=config)

    config.logger = wandb
    config.logger.run.save()
    config.run_name = config.logger.run.name  # handle name change in wandb
    atexit.register(sync_before_exit, config, wandb)

    # Data loading
    config.train_subjects = [1, 5, 6, 7, 8]
    config.val_subjects = [9, 11]
    train_loader = train_dataloader(config)
    val_loader = val_dataloader(config)

    # combinations of Encoder, Decoder to train in each epoch
    variant_dic = {
        1: [['2d', '3d'], ['rgb', 'rgb']],
        2: [['2d', '3d']],
        3: [['rgb', '3d']],
        4: [['rgb', 'rgb'], ['2d', '3d'], ['rgb', '3d']],
        5: [['2d', '3d']],
    }

    variant = variant_dic[config.variant]

    # NOTE: Naming
    # 'model' (['2d','3d'])- a list of 'net's/nn.Module in a variant is one full model (input to output),
    # 'models' - dictionary of all nets
    # -- only 1 instance of a particular 'net' is created i.e identical net types share the weights
    # -- All 'model's in 'models' share the same latent space

    models = train_utils.get_models(variant, config)  # model instances
    optimizers = train_utils.get_optims(variant, models, config)  # optimer for each pair
    schedulers = train_utils.get_schedulers(optimizers)

    if config.self_supervised:
        critic = Critic()
        models['Critic'] = critic

    # For multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f'[INFO]: Using {torch.cuda.device_count()} GPUs')
        for key in models.keys():
            models[key] = torch.nn.DataParallel(models[key])

    # To CPU or GPU or TODO TPU
    for key in models.keys():
        models[key] = models[key].to(device)
        # models[key].apply(weight_init)

    # initiate all required callbacks, keep the order in mind!!!
    cb = CallbackList([ModelCheckpoint(),
                       Logging(),
                       BetaScheduler(config, strategy="cycling")],)
    # Analyze("northern-snowflake-1584", 500),)

    cb.setup(config=config, models=models, optimizers=optimizers,
             train_loader=train_loader, val_loader=val_loader, variant=variant)

    config.mpjpe_min = float('inf')
    config.mpjpe_at_min_val = float('inf')

    # Training
    for epoch in range(1, config.epochs+1):
        for n_pair, pair in enumerate(variant):

            # VAE specific players
            vae_type = "_2_".join(pair)

            # model -- encoder, decoder / critic
            model = [models[f"Encoder{pair[0].upper()}"],
                     models[f"Decoder{pair[1].upper()}"]]

            if config.self_supervised:
                model.append(models['Critic'])

            optimizer = optimizers[n_pair]
            scheduler = schedulers[n_pair]
            config.logger.log(
                {f"{vae_type}_LR": optimizer.param_groups[0]['lr']})

            # TODO init criterion once with .to(cuda)
            training_epoch(config, cb, model, train_loader,
                           optimizer, epoch, vae_type)

            val_loss = validation_epoch(
                config, cb, model, val_loader, epoch, vae_type)

            if val_loss != val_loss:
                print("[INFO]: NAN loss")
                break

            # TODO have different learning rates for all variants
            # TODO exponential blowup of val loss and mpjpe when lr is lower than order of -9
            scheduler.step(val_loss)

            cb.on_epoch_end(config=config, val_loss=val_loss, model=model,
                            n_pair=n_pair, optimizers=optimizers, epoch=epoch)

        # TODO add better metric log for every batch with partial epoch for batch size independence
        config.logger.log({"epoch": epoch})

        if val_loss != val_loss:
            print("[INFO]: NAN loss")
            break

        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("[INFO]: LR < 1e-6. Stop training")
            break

    # sync config with wandb for easy experiment comparision
    config.logger = None  # wandb cant have objects in its config
    wandb.config.update(config)


def sync_before_exit(config, wandb):
    print("[INFO]: Sync wandb before terminating")
    config.logger = None  # wandb cant have objects in its config
    wandb.config.update(config)


def training_specific_args():

    parser = ArgumentParser()

    # training specific
    parser.add_argument('--self_supervised', default=True, type=bool,
                        help='training strategy')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--fast_dev_run', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='run all methods once to check integrity, not implemented!')
    parser.add_argument('--resume_run', default="None", type=str,
                        help='wandb run name to resume training using the saved checkpoint')
    # model specific
    parser.add_argument('--variant', default=2, type=int,
                        help='choose variant, the combination of VAEs to be trained')
    parser.add_argument('--latent_dim', default=50, type=int,
                        help='dimensions of the cross model latent space')
    parser.add_argument('--beta_warmup_epochs', default=10, type=int,
                        help='KLD weight warmup time. weight is 0 during this period')
    parser.add_argument('--beta_annealing_epochs', default=40, type=int,
                        help='KLD weight annealing time')
    parser.add_argument('--learning_rate', default=4e-4, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--pretrained', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use pretrained weights for RGB encoder')
    parser.add_argument('--train_last_block', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='train last convolution block of the RGB encoder while rest is pre-trained')
    parser.add_argument('--n_joints', default=16, type=int,
                        help='number of joints to encode and decode')
    # pose data
    parser.add_argument('--annotation_file', default=f'h36m17', type=str,
                        help='prefix of the annotation h5 file: h36m17 or debug_h36m17')
    parser.add_argument('--annotation_path', default=None, type=str,
                        help='if none, checks data folder. Use if data is elsewhere for colab/kaggle')
    # image data
    parser.add_argument('--image_path', default=f'{os.getenv("HOME")}/lab/HPE_datasets/h36m/', type=str,
                        help='path to image folders with subject action etc as folder names')
    parser.add_argument('--ignore_images', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='when true, do not load images for training')
    # output
    parser.add_argument('--save_dir', default=f'{os.path.dirname(os.path.abspath(__file__))}/checkpoints', type=str,
                        help='path to save checkpoints')
    parser.add_argument('--exp_name', default=f'run_1', type=str,
                        help='name of the current run, used to id checkpoint and other logs')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='# of batches to wait before logging training status')
    # device
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='enable cuda if available')
    parser.add_argument('--pin_memory', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='pin memory to device')
    parser.add_argument('--seed', default=400, type=int,
                        help='random seed')

    return parser


if __name__ == "__main__":
    main()
