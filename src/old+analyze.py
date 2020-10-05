import os
import sys
from argparse import ArgumentParser
import gc
import numpy as np

import torch
import wandb

from viz import plot_diff, plot_diffs, plot_umap, plot_3d
import dataloader
import train_utils
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

    val_subset = val_loader.dataset

    # Could also just import H36M Dataset instead
    val_subset = torch.utils.data.Subset(val_loader.dataset,
                                       np.random.randint(0, val_loader.dataset.__len__(), 500))

    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=None,
        shuffle=False
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
    models = train_utils.get_models(variants, config)

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
            images = []
            for t in target:
                image_ = plot_3d(np.asarray(t))
                images.append(image_)
            images = torch.cat(images,0)
            # images = np.stack(images, axis=0)
            # label_img = torch.rand(30, 3, 10, 32)
            # for i in range(30):
            #     label_img[i]*=i/100.0

            writer.add_embedding(z, metadata=action, label_img=images)
            
            # for x in range(recon.shape[0]):
                # print(recon.shape, recon[x].shape)
            
            # plot_diffs(recon, target, torch.mean(PJPE(recon, target), dim=1), grid=5)
            # print(recon[0])
            # plot_umap(z, action)

            
        # Latent Space Sampling
        # if epoch % manifold_interval == 0:
        # sample_manifold(config, model)

        del recon, target
        gc.collect()
