from model import LinearModel, weight_init

import os
import sys
from argparse import ArgumentParser
import gc
import torch
import wandb
import h5py
sys.path.insert(0, f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')

import utils
import dataloader
from processing import denormalize
from models.pose_models import PJPE

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
        os.environ['WANDB_TAGS'] = 'CPU'

    wandb.init(anonymous='allow', project="hpe3d")
    config.logger = wandb

    # prints after init, so its logged in wandb
    print(f'[INFO]: using device: {device}')

    # Data loading
    config.train_subjects = [1, 5, 6, 7, 8]
    config.val_subjects = [9, 11]

    train_loader = dataloader.train_dataloader(config)
    val_loader = dataloader.val_dataloader(config)

    model = LinearModel()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel()
                                                 for p in model.parameters()) / 1000000.0))
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    # For multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f'[INFO]: Using {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)

    # To CPU or GPU or TODO TPU
    model.to(device)

    print(f'[INFO]: Start training procedure')
    val_loss_min = float('inf')

    # Training
    config.step = 0
    for epoch in range(1, config.epochs+1):
        # training
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).float()

            optimizer.zero_grad()

            inp = batch['pose2d'].view(-1, 2*16)
            target = batch['pose3d']

            output = model(inp)
            output = output.view(target.shape)

            loss = criterion(output, target)

            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(batch['pose2d']),
                len(train_loader.dataset), 100. *
                batch_idx / len(train_loader),
                loss.item()))

            config.logger.log({"loss": loss.item()})

            loss.backward()
            optimizer.step()

        # eval
        evaluate_poses(config, model, val_loader, epoch)

    # sync config with wandb for easy experiment comparision
    config.logger = None  # wandb cant have objects in its config
    wandb.config.update(config)


def evaluate_poses(config, model, val_loader, epoch):

    ann_file_name = config.annotation_file.split('/')[-1].split('.')[0]
    norm_stats = h5py.File(
        f"{os.path.dirname(os.path.abspath(__file__))}/../data/norm_stats_{ann_file_name}_911.h5", 'r')

    model.eval()

    pjpes = []
    n_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).float()

            inp = batch['pose2d'].view(-1, 2*16)
            target = batch['pose3d']
            output = model(inp)
            output = output.view(target.shape)

            # de-normalize data to original positions
            output = denormalize(
                output,
                torch.tensor(norm_stats['mean3d'], device=config.device),
                torch.tensor(norm_stats['std3d'], device=config.device))
            target = denormalize(
                target,
                torch.tensor(norm_stats['mean3d'], device=config.device),
                torch.tensor(norm_stats['std3d'], device=config.device))

            # since the MPJPE is computed for 17 joints with roots aligned i.e zeroed
            # Not very fair, but the average is with 17 in the denom!
            output = torch.cat(
                (torch.zeros(output.shape[0], 1, 3, device=config.device), output), dim=1)
            target = torch.cat(
                (torch.zeros(target.shape[0], 1, 3, device=config.device), target), dim=1)

            pjpe = PJPE(output, target)
            # TODO plot and save samples for say each action to see improvement
            # plot_diff(output[0].numpy(), target[0].numpy(), torch.mean(mpjpe).item())
            pjpes.append(torch.sum(pjpe, dim=0))
            n_samples += pjpe.shape[0]  # to calc overall mean

    # mpjpe = torch.stack(pjpes, dim=0).mean(dim=0)
    mpjpe = torch.stack(pjpes, dim=0).sum(dim=0)/n_samples
    avg_mpjpe = torch.mean(mpjpe).item()

    config.logger.log({"MPJPE_AVG": avg_mpjpe})

    print(f' * Mean MPJPE * : {round(avg_mpjpe,4)} \n {mpjpe}')

    del pjpes
    del mpjpe
    norm_stats.close()
    del norm_stats
    gc.collect()


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
    parser.add_argument('--annotation_file', default=f'debug_h36m17', type=str,
                        help='prefix of the h5 file containing poses and camera data')
    parser.add_argument('--annotation_path', default=None, type=str,
                        help='if none, checks data folder. Use if data is elsewhere for colab/kaggle')
    parser.add_argument('--image_path', default=f'/home/datta/lab/HPE_datasets/h36m/', type=str,
                        help='path to image folders with subject action etc as folder names')
    parser.add_argument('--ignore_images', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='when true, do not load images for training')
    # training specific
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=100, type=int,
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
