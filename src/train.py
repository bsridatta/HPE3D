import atexit
import gc
import math
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

import dataloader
import train_utils
import wandb
from models import PJPE, weight_init
from trainer import training_epoch, validation_epoch
from viz import plot_diff, plot_diffs


def main():
    # Experiment Configuration`
    parser = training_specific_args()

    # Config is distributed to all the other modules
    config = parser.parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # log intervals
    eval_interval = 1  # interval to get MPJPE of 3d decoder
    manifold_interval = 1  # interval to visualize encoding in manifold

    # GPU setup
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.device = device  # Adding device to config, not already in argparse
    config.num_workers = 4 if use_cuda else 4  # for dataloader

    # wandb for experiment monitoring
    os.environ['WANDB_NOTES'] = 'divide by mean distance no norm'
    # ignore when debugging on cpu
    if not use_cuda:
        os.environ['WANDB_MODE'] = 'dryrun' # Doesnt auto sync to project
        os.environ['WANDB_TAGS'] = 'CPU'
        wandb.init(anonymous='allow', project="to_delete", config=config)
    else:
        # os.environ['WANDB_MODE'] = 'dryrun'
        wandb.init(anonymous='allow', project="hpe3d", config=config)

    config.logger = wandb
    config.logger.run.save()
    # To id weights even after changing run name
    config.run_name = config.logger.run.name
    # prints after init, so its logged in wandb
    print(f'[INFO]: using device: {device}')

    # Data loading
    config.train_subjects = [1, 5, 6, 7, 8]
    config.val_subjects = [9, 11]

    train_loader = dataloader.train_dataloader(config)
    val_loader = dataloader.val_dataloader(config)

    # combinations of Encoder, Decoder to train in an epoch
    variant_dic = {
        1: [['2d', '3d'], ['rgb', 'rgb']],
        2: [['2d', '3d']],
        3: [['rgb', '3d']],
        4: [['rgb', 'rgb'], ['2d', '3d'], ['rgb', '3d']]}

    variants = variant_dic[config.variant]

    # Intuition: Each variant is one model,
    # except they use the same weights and same latent_dim
    models = train_utils.get_models(variants, config)
    optimizers = train_utils.get_optims(models, config)
    schedulers = train_utils.get_schedulers(optimizers)

    # For multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f'[INFO]: Using {torch.cuda.device_count()} GPUs')
        for vae in range(len(models)):
            models[vae][0] = torch.nn.DataParallel(models[vae][0])
            models[vae][1] = torch.nn.DataParallel(models[vae][1])

    # To CPU or GPU or TODO TPU
    for vae in range(len(models)):
        models[vae][0].to(device)
        models[vae][1].to(device)

        # models[vae][0].apply(weight_init)
        # models[vae][1].apply(weight_init)
        
        config.logger.watch(models[vae][0], log='all')
        config.logger.watch(models[vae][1], log='all')

    # Resume training
    if config.resume_run not in "None":
        for vae in range(len(models)):
            for model_ in models[vae]:
                state = torch.load(
                    f'{config.save_dir}/{config.resume_run}_{model_.name}.pt', map_location=device)
                print(
                    f'[INFO] Loaded Checkpoint {config.resume_run}: {model_.name} @ epoch {state["epoch"]}')
                model_.load_state_dict(state['model_state_dict'])
                optimizers[vae].load_state_dict(state['optimizer_state_dict'])
                # TODO load optimizer state seperately w.r.t variant

    print(f'[INFO]: Start training procedure')
    wandb.save(
        f"{os.path.dirname(os.path.abspath(__file__))}/models/pose*")

    config.val_loss_min = float('inf')
    config.mpjpe_min = float('inf')
    config.mpjpe_at_min_val = float('inf')
    
    config.beta = 0

    # Training
    for epoch in range(1, config.epochs+1):
        for variant in range(len(variants)):
            # Variant specific players
            vae_type = "_2_".join(variants[variant])

            # model -- tuple of encoder decoder
            model = models[variant]
            optimizer = optimizers[variant]
            scheduler = schedulers[variant]
            config.logger.log(
                {f"{vae_type}_LR": optimizer.param_groups[0]['lr']})

            # Train
            # TODO init criterion once with .to(cuda)
            training_epoch(config, model, train_loader,
                           optimizer, epoch, vae_type)
            # Validation
            val_loss, recon, target, z, z_attr = validation_epoch(
                config, model, val_loader, epoch, vae_type)

            if val_loss != val_loss:
                print("[INFO]: NAN loss")
                break

            # Evaluate Performance
            if variants[variant][1] == '3d' and epoch % eval_interval == 0:
                pjpe = torch.mean(PJPE(recon, target), dim=0)
                mpjpe = torch.mean(pjpe).item()
                print(f'{vae_type} - * MPJPE * : {round(mpjpe,4)} \n {pjpe}')
                wandb.log({f'{vae_type}_mpjpe': mpjpe})
                if mpjpe < config.mpjpe_min:
                    config.mpjpe_min = mpjpe
                import viz
                viz.plot_diffs(recon[1:3].cpu(), target[1:3].cpu(), pjpe[1:3].cpu())
            # Latent Space Sampling
            # if epoch % manifold_interval == 0:
            # sample_manifold(config, model)

            del recon, target, z, z_attr
            gc.collect()

            # Model Chechpoint
            if use_cuda:
                train_utils.model_checkpoint(
                    config, val_loss, mpjpe, model, optimizer, epoch)

            # TODO have different learning rates for all variants
            # TODO exponential blowup of val loss and mpjpe when lr is lower than order of -9
            scheduler.step(val_loss)

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

# def exit_handler(config, wandb):
#     print("[INFO]: Sync wandb before terminating")

#     config.logger = None  # wandb cant have objects in its config
#     wandb.config.update(config)

def training_specific_args():

    parser = ArgumentParser()

    # training specific
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
    parser.add_argument('--latent_dim', default=500, type=int,
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
