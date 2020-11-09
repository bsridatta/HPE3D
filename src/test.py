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
from src.models import PJPE, kaiming_init, Critic
from src.trainer import validation_epoch, _validation_step
from src.callbacks import CallbackList, ModelCheckpoint, Logging, WeightScheduler, Analyze, MaxNorm
from collections import OrderedDict, defaultdict
from src.processing import post_process, random_rotate, project_3d_to_2d


def main():
    config = do_setup()

    # Data loading
    config.val_subjects = [9, 11]
    val_loader = val_dataloader(config, shuffle=False)

    variant = [['2d', '3d']]

    models = train_utils.get_models(variant, config)  # model instances
    if config.self_supervised:
        critic = Critic()
        models['Critic'] = critic
    optimizers = train_utils.get_optims(variant, models, config)  # optimer for each pair
    schedulers = train_utils.get_schedulers(optimizers)

    # For multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f'[INFO]: Using {torch.cuda.device_count()} GPUs')
        for key in models.keys():
            models[key] = torch.nn.DataParallel(models[key])

    # To CPU or GPU or TODO TPU
    for key in models.keys():
        models[key] = models[key].to(config.device)
        if key == 'Critic':
            models[key].apply(kaiming_init)

    # initiate all required callbacks, keep the order in mind!!!
    cb = CallbackList([ModelCheckpoint(),
                       Logging(),
                       WeightScheduler(config, strategy="beta_cycling"),
                       #    MaxNorm()
                       ])

    cb.setup(config=config, models=models, optimizers=optimizers,
             val_loader=val_loader, variant=variant)

    config.mpjpe_min = float('inf')
    config.mpjpe_at_min_val = float('inf')

    n_pair = 1
    pair = variant[0]
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

    ####################################################################

    bh = False
    epochs = 10
    missing_joints = 0
    save = True

    if bh:
        zv = False
    else:
        zv = True

    cb.on_validation_start()

    normalize_pose = True
    n_pjpes = []

    if zv:
        epochs = 1

    for epoch in range(epochs):

        t_data = defaultdict(list)
        loss_dic = defaultdict(int)

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(config.device)

                if missing_joints:
                    pose = batch['pose2d']

                    p_limbs = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
                    p_limbs = torch.Tensor(p_limbs).to(pose.device)
                    p_limbs = p_limbs.repeat(pose.shape[0], 1)

                    miss_idx = torch.multinomial(p_limbs, missing_joints, replacement=False)
                    for i in range(missing_joints):
                        pose[torch.arange(pose.shape[0]), miss_idx[:, i], :] = 0

                    batch['pose2d'] = pose  # not needed

                output = _validation_step(batch, batch_idx, model, epoch, config, eval=zv)

                loss_dic['loss'] += output['loss'].item()
                loss_dic['recon_loss'] += output['log']['recon_loss'].item()
                loss_dic['kld_loss'] += output['log']['kld_loss'].item()

                if config.self_supervised:
                    loss_dic['gen_loss'] += output['log']['gen_loss'].item()
                    loss_dic['critic_loss'] += output['log']['critic_loss'].item()
                    loss_dic['D_x'] += output['log']['D_x']
                    loss_dic['D_G_z1'] += output['log']['D_G_z1']
                    loss_dic['D_G_z2'] += output['log']['D_G_z2']

                for key in output['data'].keys():
                    t_data[key].append(output['data'][key])

                del output
                gc.collect()

        avg_loss = loss_dic['loss']/len(val_loader)  # return for scheduler

        for key in t_data.keys():
            t_data[key] = torch.cat(t_data[key], 0)

        # performance
        t_data['recon_3d_org'] = t_data['recon_3d'].detach()
        if '3D' in model[1].name:
            if normalize_pose and not config.self_supervised:
                t_data['recon_3d'], t_data['target_3d'] = post_process(
                    t_data['recon_3d'], t_data['target_3d'])

            elif config.self_supervised:
                t_data['recon_3d'], t_data['target_3d'] = post_process(
                    t_data['recon_3d'].to('cpu'), t_data['target_3d'].to('cpu'),
                    self_supervised=True, procrustes_enabled=True)

            # Speed up procrustes alignment with CPU!
            t_data['recon_3d'] = t_data['recon_3d'].to(config.device)
            t_data['target_3d'] = t_data['target_3d'].to(config.device)

            pjpe_ = PJPE(t_data['recon_3d'], t_data['target_3d'])
            avg_pjpe = torch.mean((pjpe_), dim=0)
            avg_mpjpe = torch.mean(avg_pjpe).item()
            pjpe = torch.mean(pjpe_, dim=1)

            actions = t_data['action']
            mpjpe_pa = {}  # per action
            for i in torch.unique(actions):
                res = torch.mean(pjpe[actions == i])
                mpjpe_pa[i.item()] = res.item()

            n_pjpes.append(pjpe)

            config.logger.log({"pjpe": pjpe.cpu()})

            # average epochs output
            avg_output = {}
            avg_output['log'] = {}

            avg_output['loss'] = loss_dic['loss']/len(val_loader)
            avg_output['log']['recon_loss'] = loss_dic['recon_loss']/len(val_loader)
            avg_output['log']['kld_loss'] = loss_dic['kld_loss']/len(val_loader)

            # print to console
            print(f"{vae_type} Validation:",
                  f"\t\tLoss: {round(avg_output['loss'],4)}",
                  f"\tReCon: {round(avg_output['log']['recon_loss'], 4)}",
                  f"\tKLD: {round(avg_output['log']['kld_loss'], 4)}", end='')

            print(f"\n MPJPE: {avg_mpjpe} \n {avg_pjpe} \n")

            print(f"PJPE 1 {pjpe[0]} \n")

    if bh:
        # multi-hypothesis
        mh = torch.stack(n_pjpes)

        # mpjpe for random z
        mpjpe_z = torch.mean(mh, dim=1)
        print("mpjpe for random z \n", mpjpe_z)

        mpjpe_bh = torch.min(mh, dim=0).values.mean()
        print("mpjpe best hypothesis: ", mpjpe_bh)

    if zv:
        print(f"\n ZV MPJPE: {avg_mpjpe} \n {avg_pjpe} \n")

    if save:
        torch.save(t_data, f"src/results/t_data_{config.resume_run}_bh_{bh}_mj_{missing_joints}.pt")

    viz.mpl_plots.plot_errors(t_data['recon_3d'].cpu().numpy(),
                              t_data['target_3d'].cpu().numpy(),
                              torch.mean(PJPE(t_data['recon_3d'].cpu(),
                                              t_data['target_3d'].cpu()), dim=1),
                              grid=5)


def do_setup():
    # Experiment Configuration, Config, is distributed to all the other modules
    parser = training_specific_args()
    config = parser.parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # GPU setup
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.device = device  # Adding device to config, not already in argparse
    config.num_workers = 4 if use_cuda else 4  # for dataloader

    # wandb for experiment monitoring
    os.environ['WANDB_TAGS'] = 'inference'
    os.environ['WANDB_NOTES'] = 'inference'

    if not config.wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    # ignore when debugging on cpu
    if not use_cuda:
        # os.environ['WANDB_MODE'] = 'dryrun'  # Doesnt auto sync to project
        os.environ['WANDB_TAGS'] = 'CPU'
        wandb.init(anonymous='allow', project="hpe3d", config=config)  # to_delete
    else:
        wandb.init(anonymous='allow', project="hpe3d", config=config)

    config.logger = wandb
    config.logger.run.save()
    config.run_name = config.logger.run.name  # handle name change in wandb

    return config


def training_specific_args():

    parser = ArgumentParser()

    # training specific
    parser.add_argument('--self_supervised', default=True, type=bool,
                        help='training strategy')
    parser.add_argument('--epochs', default=800, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=2560, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--fast_dev_run', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='run all methods once to check integrity, not implemented!')
    parser.add_argument('--resume_run', default="absurd-music-3244", type=str,
                        help='wandb run name to resume training using the saved checkpoint')
    parser.add_argument('--test', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='run validatoin epoch only')
    # model specific
    parser.add_argument('--variant', default=2, type=int,
                        help='choose variant, the combination of VAEs to be trained')
    parser.add_argument('--latent_dim', default=51, type=int,
                        help='dimensions of the cross model latent space')
    parser.add_argument('--critic_weight', default=1e-3, type=float,
                        help='critic weight for self supervised procedure')
    parser.add_argument('--critic_annealing_epochs', default=10, type=int,
                        help='critic weight annealing time')
    parser.add_argument('--beta_warmup_epochs', default=10, type=int,
                        help='KLD weight warmup time. weight is 0 during this period')
    parser.add_argument('--beta_annealing_epochs', default=40, type=int,
                        help='KLD weight annealing time')
    parser.add_argument('--noise_level', default=0.0, type=float,  # 0.01
                        help='percentage of noise to inject for critic training')
    parser.add_argument('--beta_max', default=0.01, type=float,  # 0.01
                        help='maximum value of beta during annealing or cycling')
    parser.add_argument('--learning_rate', default=2e-4, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--pretrained', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use pretrained weights for RGB encoder')
    parser.add_argument('--train_last_block', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='train last convolution block of the RGB encoder while rest is pre-trained')
    parser.add_argument('--n_joints', default=16, type=int,
                        help='number of joints to encode and decode')
    parser.add_argument('--p_miss', default=0.2, type=int,
                        help='number of joints to encode and decode')
    # pose data
    parser.add_argument('--annotation_file', default=f'h36m17', type=str,
                        help='prefix of the annotation h5 file: h36m17 or h36m17_2 or debug_h36m17')
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
    parser.add_argument('--wandb', type=bool, default=False,
                        help='wandb')
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
