
import logging
import os
import sys
from argparse import ArgumentParser

import torch
from torch.utils.tensorboard import SummaryWriter

import dataloader
import utils
from trainer import training_epoch, validation_epoch


def main():
    # Experiment Configuration
    parser = training_specific_args()
    # Config would be distributed to all the modules
    config = parser.parse_args()
    torch.manual_seed(config.seed)
    logging.getLogger().setLevel(logging.INFO)

    # Tensorboard Logs
    suffix = 0
    while os.path.exists(f"../logs/{config.exp_name}_{suffix}"):
        suffix += 1
    writer = SummaryWriter(f"../logs/{config.exp_name}_{suffix}")
    config.writer = writer

    # GPU setup
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f'using device: {device}')
    config.device = device  # Adding device to config, not already in argparse
    config.num_workers = 4 if use_cuda else 4  # for dataloader
    config.pin_memory = False if use_cuda else False
    # easier to know what params are used in the runs
    writer.add_text("config", str(config))

    # Data loading
    config.train_subjects = [1, 5, 6, 7, 8]
    config.val_subjects = [9, 11]

    train_loader = dataloader.train_dataloader(config)
    val_loader = dataloader.val_dataloader(config)
    # test_loader = dataloader.test_dataloader(config)

    # combinations of Encoder, Decoder to train in an epoch
    variant_dic = {
        1: [['2d', '3d'], ['rgb', '3d']]
    } 
    variants = variant_dic[1]
    # Intuition: Each variant is one model,
    # except they use the same weights and same latent_dim
    models = utils.get_models(variants, config)
    optimizers = utils.get_optims(models, config)
    schedulers = utils.get_schedulers(optimizers)

    # For multiple GPUs
    if torch.cuda.device_count() > 1:
        logging.info(f'Using {torch.cuda.device_count()} GPUs')
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

    logging.info('Start training procedure')
    val_loss_min = float('inf')

    # Training
    config.step = 0
    for epoch in range(1, config.epochs+1):
        for variant in range(len(variants)):
            # Variant specific players
            model = models[variant]  # tuple of encoder decoder
            optimizer = optimizers[variant]
            scheduler = schedulers[variant]

            # Train
            # TODO convert batch to tensors

            training_epoch(config, model, train_loader, optimizer, epoch)
            # val_loss = validation_epoch(config, model, val_loader)
            # scheduler.step(val_loss) # TODO verify if it knows the right optimizer
            print("var", variants[variant])

        # if val_loss < val_loss_min:
        #     val_loss_min = val_loss
        #     try:
        #         state_dict = model.module.state_dict()
        #     except AttributeError:
        #         state_dict = model.state_dict()

        #     state = {
        #         'epoch': epoch,
        #         'val_loss': val_loss,
        #         'model_state_dict': state_dict,
        #         'optimizer_state_dict': optimizer.state_dict()
        #     }
        #     torch.save(state, f'{config.save_dir}/{config.exp_name}.pt')
        #     logging.info(f'Saved pt: {config.save_dir}/{config.exp_name}.pt')

    writer.close()


def training_specific_args():

    parser = ArgumentParser()

    # GPU
    parser.add_argument('--cuda', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--seed', default=400, type=int)

    # data
    parser.add_argument(
        '--annotation_file', default=f'data/debug2_h36m17.h5', type=str)
    parser.add_argument(
        '--image_path', default=f'../../HPE_datasets/h36m/', type=str)

    # network args
    # RGB
    parser.add_argument('--latent_dim', default=30, type=int)
    parser.add_argument('--pretrained', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_last_block', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    # Pose
    parser.add_argument('--n_joints', default=17, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)

    # training params
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--fast_dev_run', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--resume_pt', default=0, type=str)

    # output
    parser.add_argument(
        '--save_dir', default=f'{os.path.dirname(os.getcwd())}/checkpoints', type=str)
    parser.add_argument('--exp_name', default=f'run_1', type=str)
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    return parser


if __name__ == "__main__":
    main()
