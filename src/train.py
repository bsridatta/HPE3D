import atexit
import gc
import math
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import wandb

sys.path.append("../src")  # noqa

from src import train_utils
from src import viz
from src.dataloader import train_dataloader, val_dataloader
from src.models import PJPE, kaiming_init, Critic
from src.trainer import training_epoch, validation_epoch
from src.callbacks import CallbackList, ModelCheckpoint, Logging, WeightScheduler, Analyze, MaxNorm


def main():

    config = do_setup()

    # Data loading
    train_loader = train_dataloader(config)
    val_loader = val_dataloader(config)

    # TODO REMOVE
    # combinations of Encoder, Decoder to train in each epoch
    variant_dic = {
        1: [['2d', '3d'], ['rgb', 'rgb']],
        2: [['2d', '3d']],
        3: [['rgb', '3d']],
        4: [['rgb', 'rgb'], ['2d', '3d'], ['rgb', '3d']],
        5: [['2d', '3d']],
        6: [['rgb', 'rgb']],
    }

    variant = variant_dic[config.variant]

    models = train_utils.get_models(variant, config)  # model instances
    if config.self_supervised:
        critic = Critic()
        models['Critic'] = critic
    optimizers = train_utils.get_optims(
        variant, models, config)  # optimer for each pair
    schedulers = train_utils.get_schedulers(optimizers, config)

    # data parallel
    if torch.cuda.device_count() > 1:
        print(f'[INFO]: Using {torch.cuda.device_count()} GPUs')
        for key in models.keys():
            models[key] = torch.nn.DataParallel(models[key])

    # To CPU or GPU or TODO TPU
    for key in models.keys():
        models[key] = models[key].to(config.device)
        # if key == 'Critic':
        models[key].apply(kaiming_init)

    # initiate all required callbacks, keep the order in mind!!!
    cb = CallbackList([ModelCheckpoint(),
                       Logging(),
                       WeightScheduler(config, strategy="beta_cycling"),
                       #    WeightScheduler(config, strategy="noise_annealing"),
                       #    WeightScheduler(config, strategy="critic_cycling"),
                       #    MaxNorm()
                       ])

    cb.setup(config=config, models=models, optimizers=optimizers,
             train_loader=train_loader, val_loader=val_loader, variant=variant)

    config.mpjpe_min = float('inf')

    # Training
    for epoch in range(1, config.epochs+1):
        for n_pair, pair in enumerate(variant):

            # VAE specific players
            vae_type = "_2_".join(pair)

            # model -- encoder, decoder / critic
            model = [models[f"Encoder{pair[0].upper()}"],
                     models[f"Decoder{pair[1].upper()}"]]
            optimizer = [optimizers[n_pair]]
            scheduler = [schedulers[n_pair]]

            if config.self_supervised:
                model.append(models['Critic'])
                optimizer.append(optimizers[-1])
                scheduler.append(schedulers[-1])

            config.logger.log(
                {f"{vae_type}_LR": optimizer[0].param_groups[0]['lr']})

            # TODO init criterion once with .to(cuda)
            training_epoch(config, cb, model, train_loader,
                           optimizer, epoch, vae_type)

            val_loss = 0
            if epoch % config.validation_interval == 0:
                val_loss = validation_epoch(
                    config, cb, model, val_loader, epoch, vae_type)

                if val_loss != val_loss:
                    print("[WARNING]: NAN loss")
                    break

                # only model ckpt as of now - 
                # TODO change to on_validation_end as no change in mpjpe_min without validation run
                # ideally should be in validation epoch code but easier acces to model info here
                # checkpoint if new mpjpe from validation is best
                cb.on_epoch_end(config=config, val_loss=val_loss, model=model,
                                n_pair=n_pair, optimizers=optimizers, epoch=epoch)

            # TODO have different learning rates for generator and discriminator
            scheduler[0].step()
            scheduler[1].step()


        # TODO add better metric log for every batch with partial epoch for batch size independence
        config.logger.log({"epoch": epoch})

        if val_loss != val_loss:
            print("[INFO]: NAN loss")
            break

        if optimizer[0].param_groups[0]['lr'] < 1e-6:
            print("[INFO]: LR < 1e-6. Stop training")
            break
            

    # sync config with wandb for easy experiment comparision
    config.logger = None  # wandb cant have objects in its config
    wandb.config.update(config)


def do_setup():
    parser = get_argparser()
    config = parser.parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # GPU setup
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.device = device  # Adding device to config, not already in argparse

    # wandb for experiment monitoring
    os.environ['WANDB_TAGS'] = 'New_Scaling'
    # os.environ['WANDB_NOTES'] = 'None'

    # ignore when debugging on cpu
    if config.device.type == 'cpu':
        os.environ['WANDB_MODE'] = 'dryrun'  # Doesnt auto sync to project
        os.environ['WANDB_TAGS'] = 'CPU'
        wandb.init(anonymous='allow', project="hpe3d",
                   config=config)  # to_delete
    else:
        # os.environ['WANDB_MODE'] = 'dryrun'
        wandb.init(anonymous='allow', project="hpe3d", config=config)

    config.logger = wandb
    config.logger.run.save()
    config.run_name = config.logger.run.name  # handle name change in wandb
    atexit.register(sync_before_exit, config, wandb)

    return config


def sync_before_exit(config, wandb):
    print("[INFO]: Sync wandb before terminating")
    config.logger = None  # wandb cant have objects in its config
    wandb.config.update(config)


def get_argparser():

    parser = ArgumentParser()

    # training specific
    parser.add_argument('--epochs', default=2, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--fast_dev_run', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='run all methods once to check integrity')
    parser.add_argument('--resume_run', default="None", type=str,
                        help='wandb run name to resume training using the saved checkpoint')
    parser.add_argument('--test', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='run validatoin epoch only')
    parser.add_argument('--is_ss', default=True, type=bool,
                        help='training strategy - self supervised')
    parser.add_argument('--top_k', default=True, type=bool,
                        help='top k realistic samples to train generator')
    parser.add_argument('--top_k_gamma', default=0.99, type=float,
                        help='# decay rate of k')
    parser.add_argument('--top_k_min', default=0.5, type=float,
                        help='# least % of k')
    
    # model specific
    parser.add_argument('--variant', default=2, type=int,
                        help='choose variant, the combination of VAEs to be trained')
    parser.add_argument('--latent_dim', default=51, type=int,
                        help='dimensions of the cross model latent space')
    parser.add_argument('--lambda_gen', default=1, type=float,
                        help='critic weight for self supervised procedure')
    parser.add_argument('--lambda_disc', default=1e-3, type=float,
                        help='critic weight for self supervised procedure')
    parser.add_argument('--lambda_kld', default=1e-3, type=float,  # 0.01
                        help='beta (kld coeff in b-vae) or maximum value of beta during annealing or cycling')
   
    parser.add_argument('--critic_annealing_epochs', default=10, type=int,
                        help='critic weight annealing time')
    parser.add_argument('--beta_warmup_epochs', default=10, type=int,
                        help='KLD weight warmup time. weight is 0 during this period')
    parser.add_argument('--beta_annealing_epochs', default=40, type=int,
                        help='KLD weight annealing time')
    parser.add_argument('--noise_level', default=0.0, type=float, 
                        help='percentage of noise to inject for critic training')
    parser.add_argument('--flip_labels_n_e', default=0, type=int,  
                        help='flip real fake labels for critic every n epochs')
    
    parser.add_argument('--lr_gen', default=2e-4, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--lr_disc', default=2e-4, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--lr_decay', default=0.95, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--p_miss_joints', default=0.0, type=float,
                        help='emulate joint occlusion')
    # data files
    parser.add_argument('--train_file', default=f'{os.path.dirname(os.path.abspath(__file__))}/data/h36m_train_sh.h5', type=str,
                        help='abs path to training data file')
    parser.add_argument('--test_file', default=f'{os.path.dirname(os.path.abspath(__file__))}/data/h36m_test_sh.h5', type=str,
                        help='abs path to validation data file')
    # output
    # parser.add_argument('--save_dir', default=f'{os.path.dirname(os.path.abspath(__file__))}/checkpoints', type=str,
    #                     help='path to save checkpoints')
    parser.add_argument('--exp_name', default=f'run_1', type=str,
                        help='name of the current run, used to id checkpoint and other logs')
    parser.add_argument('--log_image_interval', type=int, default=1,
                        help='log images during eval epoch falling in this interval')
    parser.add_argument('--validation_interval', type=int, default=5,
                        help='validate every nth epoch')

    # device
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='enable cuda if available')
    parser.add_argument('--pin_memory', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='pin memory to device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='workers for data loader')
    parser.add_argument('--seed', default=400, type=int,
                        help='random seed')

    return parser


if __name__ == "__main__":
    main()
