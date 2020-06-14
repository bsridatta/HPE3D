import os
import sys
from argparse import ArgumentParser
import gc
import numpy as np

import torch
import wandb

from viz import plot_diff, plot_diffs, plot_umap
import dataloader
import utils
from models import PJPE
from trainer import (training_epoch,
                     validation_epoch)
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

def main():
    """
    Similar to train.py but without training code. 
    Create for visualization of latent space and pose reconstructions, without having to disturb train.py
    """
    # Experiment Configuration
    parser = training_specific_args()

    # Config is distributed to all the other modules
    config = parser.parse_args()
    torch.manual_seed(config.seed)

    # GPU setup
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.device = device  # Adding device to config, not already in argparse
    config.num_workers = 4 if use_cuda else 4  # for dataloader

    # wandb for experiment monitoring, ignore when debugging on cpu
    if not use_cuda:
        os.environ['WANDB_MODE'] = 'dryrun'
        os.environ['WANDB_TAGS'] = 'CPU'

    wandb.init(anonymous='allow', project="to_delete", sync_tensorboard=True)
    config.logger = wandb
    config.run_name = config.logger.run.name
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=wandb.run.dir)

    # prints after init, so its logged in wandb
    print(f'[INFO]: using device: {device}')

    # Data loading
    config.train_subjects = [1, 5, 6, 7, 8]
    config.val_subjects = [9, 11]

    # train_loader = dataloader.train_dataloader(config) # dont need it!
    val_loader = dataloader.val_dataloader(config)
    
    # Could also just import H36M Dataset instead
    val_subset = torch.utils.data.Subset(val_loader.dataset,
                                       np.random.randint(0, val_loader.dataset.__len__(), 500))
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=None,
        shuffle=True
    )
    print("subset samples -", len(val_loader.dataset))

    # combinations of Encoder, Decoder to train in an epoch
    variant_dic = {
        1: [['2d', '3d'], ['rgb', 'rgb']],
        2: [['2d', '3d']],
        3: [['rgb', 'rgb']],
        4: [['rgb', 'rgb'], ['2d', '3d'], ['rgb', '3d']]}

    variants = variant_dic[config.variant]

    # Intuition: Each variant is one model,
    # except they use the same weights and same latent_dim
    models = utils.get_models(variants, config)

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

    # Resume training
    if config.resume_run not in "None":
        for vae in range(len(models)):
            for model_ in models[vae]:
                state = torch.load(
                    f'{config.save_dir}/{config.resume_run}_{model_.name}.pt', map_location=device)
                model_.load_state_dict(state['model_state_dict'])
                print(
                    f'[INFO] Loaded Checkpoint {config.resume_run}: {model_.name} @ epoch: {state["epoch"]}')

    epoch = 1
    for variant in range(len(variants)):
        # Variant specific players
        vae_type = "_2_".join(variants[variant])

        # model -- tuple of encoder decoder
        model = models[variant]

        # Validation
        val_loss, recon, target, z, action = validation_epoch(
            config, model, val_loader, epoch, vae_type)

        # Evaluate Performance
        if variants[variant][1] == '3d':
            pjpe = torch.mean(PJPE(recon, target), dim=0)
            mpjpe = torch.mean(pjpe).item()
            print(f'{vae_type} - * MPJPE * : {round(mpjpe,4)} \n {pjpe}')
            wandb.log({f'{vae_type}_mpjpe': mpjpe})

            writer.add_embedding(z, metadata=action)

            # plot_diffs(recon, target, torch.mean(PJPE(recon, target), dim=1), grid=5)
            # plot_umap(z, action)

            
        # Latent Space Sampling
        # if epoch % manifold_interval == 0:
        # sample_manifold(config, model)

        del recon, target
        gc.collect()


def training_specific_args():

    parser = ArgumentParser()

    # training specific
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=30, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--fast_dev_run', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='run all methods once to check integrity, not implemented!')
    parser.add_argument('--resume_run', default="laced-puddle-297", type=str,
                        help='wandb run name to resume training using the saved checkpoint')
    # model specific
    parser.add_argument('--variant', default=2, type=int,
                        help='choose variant, the combination of VAEs to be trained')
    parser.add_argument('--latent_dim', default=100, type=int,
                        help='dimensions of the cross model latent space')
    parser.add_argument('--beta', default=0.001, type=float,
                        help='KLD weight')
    parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use pretrained weights for RGB encoder')
    parser.add_argument('--train_last_block', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='train last convolution block of the RGB encoder while rest is pre-trained')
    parser.add_argument('--n_joints', default=16, type=int,
                        help='number of joints to encode and decode')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='learning rate for all optimizers')
    # GPU
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='enable cuda if available')
    parser.add_argument('--pin_memory', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='pin memory to device')
    parser.add_argument('--seed', default=400, type=int,
                        help='random seed')
    # data
    parser.add_argument('--annotation_file', default=f'h36m17', type=str,
                        help='prefix of the annotation h5 file: h36m17, debug_h36m17 etc')
    parser.add_argument('--annotation_path', default=None, type=str,
                        help='if none, checks data folder. Use if data is elsewhere for colab/kaggle')
    parser.add_argument('--image_path', default=f'/home/datta/lab/HPE_datasets/h36m_pickles/', type=str,
                        help='path to image folders with subject action etc as folder names')
    parser.add_argument('--ignore_images', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='when true, do not load images for training')
    # output
    parser.add_argument('--save_dir', default=f'{os.path.dirname(os.path.abspath(__file__))}/checkpoints', type=str,
                        help='path to save checkpoints')
    parser.add_argument('--exp_name', default=f'run_1', type=str,
                        help='name of the current run, used to id checkpoint and other logs')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='# of batches to wait before logging training status')

    return parser


if __name__ == "__main__":
    main()
