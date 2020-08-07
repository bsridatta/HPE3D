import sys

import torch

from src.models import (PJPE, Decoder3D, DecoderRGB, Encoder2D, EncoderRGB)


def get_models(variant, config):
    '''
    get models based on model names in the variant

    Arguments:
        variant (list(list)) -- all the combination of models
        config (namespace) -- contain all params for the pipeline

    Returns:
        models (dic) -- dic of unique instances of required models 
    '''
    models = {}

    for pair in variant:
        encoder = getattr(sys.modules[__name__],
                          f"Encoder{pair[0].upper()}")
        decoder = getattr(sys.modules[__name__],
                          f"Decoder{pair[1].upper()}")
        models[f"Encoder{pair[0].upper()}"] = encoder(config.latent_dim)
        models[f"Decoder{pair[1].upper()}"] = decoder(config.latent_dim)

    return models


def get_optims(variant, models, config):
    '''
    get optimizers for models

    Arguments:
        variant (list(list)) -- all the combination of models
        models (dict) -- dict of model instances
        config (namespace) -- contain all params for the pipeline

    Returns:
        optims (list) -- one optimizer for a model [encoder, decoder]
    '''
    optims = []
    for pair in variant:
        encoder = models[f"Encoder{pair[0].upper()}"]
        decoder = models[f"Decoder{pair[1].upper()}"]
        params = list(encoder.parameters())+list(decoder.parameters())
        # TODO have specific learning rate according to combo
        optimizer = torch.optim.Adam(params, lr=config.learning_rate)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10,
                                                               factor=0.3, verbose=True, threshold=1e-5)
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
        # criterion = torch.nn.L1Loss()
        # criterion = PJPE
        criterion = torch.nn.MSELoss()
    else:
        print("MODEL NOT DEFINED")
        exit()

    return (inp, target, criterion)
