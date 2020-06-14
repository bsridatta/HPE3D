import sys

import torch

from models import (Decoder3D, DecoderRGB,
                    Encoder2D, EncoderRGB,
                    image_recon_loss, PJPE)


def get_models(variants, config):
    '''
    get models based on variant name
    Model -> pair of Encoder and Decoder

    Arguments:
        variants (list(list)) -- list of [encoder_type, decoder_type] pairs
        config (namespace) -- contain all params for the pipeline

    Returns:
        models (list(list)) -- list of VAE [encoder_object, decoder_object]
    '''
    models = []

    for variant in variants:
        encoder = getattr(sys.modules[__name__],
                          f"Encoder{variant[0].upper()}")
        decoder = getattr(sys.modules[__name__],
                          f"Decoder{variant[1].upper()}")
        models.append([encoder(config.latent_dim), decoder(config.latent_dim)])

    return models


def get_optims(models, config):
    '''
    get optimizers for models

    Arguments:
        models (list(list)) -- list of VAE [encoder_object, decoder_object]
        config (namespace) -- contain all params for the pipeline

    Returns:
        optims (list) -- one optimizer for an enoceder and decoder pair
    '''
    optims = []
    for encoder, decoder in models:
        model_params = list(encoder.parameters())+list(decoder.parameters())
        optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)
        optims.append(optimizer)
    return optims


def get_schedulers(optimizers):
    '''
    get scheduler for each optimizer

    Arguments:
        optims (list) -- one optimizer for an enoceder and decoder pair

    Returns:
        optims (list) -- one optimizer for an enoceder and decoder pair

    '''
    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5,
                                                               factor=0.3, verbose=True)
        schedulers.append(scheduler)

    return schedulers


def get_inp_target_criterion(encoder, decoder, batch):
    '''
    get appropriate data and training modules current VAE
    based on if the i/o are RGB, 2D or 3D

    Arguments:
        encoder (obj) -- encoder object of the current variation
        decoder (obj) -- decoder object of the current variation
        batch (dic) -- current batch data

    Returns:
        input (list) -- input batch
        target (list) -- taget batch
        criterion (obj) -- respective loss function

    '''
    if 'RGB' in encoder.__class__.__name__:
        inp = batch['image'].float()
    elif '2D' in encoder.__class__.__name__:
        inp = batch['pose2d'].float()
    elif '3D' in encoder.__class__.__name__:
        inp = batch['pose3d'].float()
    else:
        print("MODEL NOT DEFINED")
        exit()

    if 'RGB' in decoder.__class__.__name__:
        target = batch['image'].float()
        # criterion = torch.nn.BCELoss()
        criterion = torch.nn.L1Loss()
    elif '2D' in decoder.__class__.__name__:
        target = batch['pose2d'].float()
        criterion = torch.nn.L1Loss()
    elif '3D' in decoder.__class__.__name__:
        target = batch['pose3d'].float()
        # criterion = PJPE
        criterion = torch.nn.MSELoss()
    else:
        print("MODEL NOT DEFINED")
        exit()

    return (inp, target, criterion)

def model_checkpoint(config, val_loss, model, optimizer, epoch):
    if val_loss < config.val_loss_min:
        config.val_loss_min = val_loss

        for model_ in model:
            try:
                state_dict = model_.module.state_dict()
            except AttributeError:
                state_dict = model_.state_dict()

            state = {
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict()
            }
            # TODO save optimizer state seperately
            torch.save(state, f'{config.save_dir}/{config.logger.run.name}_{model_.name}.pt')
            config.logger.save(f'{config.save_dir}/{config.logger.run.name}_{model_.name}.pt')
            print(f'[INFO] Saved pt: {config.save_dir}/{config.logger.run.name}_{model_.name}.pt')

            del state