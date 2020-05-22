import os
import sys
from argparse import ArgumentParser

import torch
import wandb

import dataloader
import utils
from trainer import (evaluate_poses, sample_manifold, training_epoch,
                     validation_epoch)


def main():
    # Experiment Configuration
    parser = training_specific_args()

    # Config is distributed to all the other modules
    config = parser.parse_args()
    torch.manual_seed(config.seed)

    # log intervals
    eval_interval = 1  # interval to get MPJPE of 3d decoder
    manifold_interval = 1  # interval to visualize encoding in manifold

    # GPU setup
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.device = device  # Adding device to config, not already in argparse
    config.num_workers = 4 if use_cuda else 4  # for dataloader

    # wandb for experiment monitoring, ignore when debugging on cpu
    if not use_cuda:
        os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(anonymous='allow', project="hpe3d")
    config.logger = wandb

    # prints after init, so its logged in wandb
    print(f'[INFO]: using device: {device}') 

    # Data loading
    config.train_subjects = [1, 5, 6, 7, 8]
    config.val_subjects = [9, 11]

    # config.train_subjects = [1, 5]
    # config.val_subjects = [1, 5, 6, 7, 8, 9, 11]

    train_loader = dataloader.train_dataloader(config)
    val_loader = dataloader.val_dataloader(config)

    # combinations of Encoder, Decoder to train in an epoch
    variant_dic = {
        1: [['2d', '3d'], ['rgb', 'rgb']],
        2: [['2d', '3d']],
        3: [['rgb', 'rgb']]}
    variants = variant_dic[config.variant]

    # Intuition: Each variant is one model,
    # except they use the same weights and same latent_dim
    models = utils.get_models(variants, config)
    optimizers = utils.get_optims(models, config)
    schedulers = utils.get_schedulers(optimizers)

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
    # if config.resume_pt:
    #     logging.info(f'Loading {config.resume_pt}')
    #     state = torch.load(f'{config.save_dir}/{config.resume_pt}')
    #     model.load_state_dict(state['state_dict'])
    #     optimizer.load_state_dict(state['optimizer'])

    print(f'[INFO]: Start training procedure')
    val_loss_min = float('inf')

    # Training
    config.step = 0
    for epoch in range(1, config.epochs+1):
        for variant in range(len(variants)):
            # Variant specific players
            vae_type = "_2_".join(variants[variant])

            # model -- tuple of encoder decoder
            model = models[variant]
            optimizer = optimizers[variant]
            scheduler = schedulers[variant]
            config.logger.log({f"{vae_type}_LR": optimizer.param_groups[0]['lr']})
            # TODO add bad epochs
            
            # Train
            # TODO init criterion once with .to(cuda)
            training_epoch(config, model, train_loader,
                           optimizer, epoch, vae_type)

            # Validation
            # val_loss = validation_epoch(
            #     config, model, val_loader, epoch, vae_type)

            # Latent Space Sampling 
            # TODO itegrate with evaluate_poses
            # if epoch % manifold_interval == 0:
            # sample_manifold(config, model)

            # Evaluate Performance
            # if variants[variant][1] == '3d' and epoch % eval_interval == 0:
            #     evaluate_poses(config, model, val_loader, epoch, vae_type)

            # TODO have different learning rates for all variants
            # TODO exponential blowup of val loss and mpjpe when lr is lower than order of -9
            # scheduler.step(val_loss)

    # sync config with wandb for easy experiment comparision
    config.logger = None  # wandb cant have objects in its config
    wandb.config.update(config)


def training_specific_args():

    parser = ArgumentParser()

    # GPU
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='enable cuda if available')
    parser.add_argument('--pin_memory', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='pin memory to device')
    parser.add_argument('--seed', default=400, type=int,
                        help='random seed')
    # data
    parser.add_argument('--annotation_file', default=f'h36m17', type=str,
                        help='prefix of the annotation h5 file: h36m17 or debug_h36m17')
    parser.add_argument('--annotation_path', default=None, type=str,
                        help='if none, checks data folder. Use if data is elsewhere for colab/kaggle')
    parser.add_argument('--image_path', default=f'/home/datta/lab/HPE_datasets/h36m/', type=str,
                        help='path to image folders with subject action etc as folder names')
    parser.add_argument('--ignore_images', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='when true, do not load images for training')
    # training specific
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=3, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--fast_dev_run', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='run all methods once to check integrity, not implemented!')
    parser.add_argument('--resume_pt', default=0, type=str,
                        help='resume training using the saved checkpoint')
    # model specific
    parser.add_argument('--variant', default=2, type=int,
                        help='choose variant, the combination of VAEs to be trained')
    parser.add_argument('--latent_dim', default=100, type=int,
                        help='dimensions of the cross model latent space')
    parser.add_argument('--beta', default=1, type=float,
                        help='KLD weight')
    parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use pretrained weights for RGB encoder')
    parser.add_argument('--train_last_block', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='train last convolution block of the RGB encoder while rest is pre-trained')
    parser.add_argument('--n_joints', default=16, type=int,
                        help='number of joints to encode and decode')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='learning rate for all optimizers')
    # output
    parser.add_argument('--save_dir', default=f'{os.path.dirname(os.getcwd())}/checkpoints', type=str,
                        help='path to save checkpoints')
    parser.add_argument('--exp_name', default=f'run_1', type=str,
                        help='name of the current run, used to id checkpoint and other logs')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='# of batches to wait before logging training status')

    return parser


if __name__ == "__main__":
    main()
