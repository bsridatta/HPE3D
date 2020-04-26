import sys

import torch

from models import (Decoder3D, DecoderRGB, Encoder2D, Encoder3D, EncoderRGB)


def get_models(variants, config):
    models = []
    for variant in variants:
        encoder = getattr(sys.modules[__name__],
                          f"Encoder{variant[0].upper()}")
        decoder = getattr(sys.modules[__name__],
                          f"Decoder{variant[1].upper()}")
        models.append([encoder(config.latent_dim), decoder(config.latent_dim)])
    return models


def get_optims(models, config):
    optims = []
    for encoder, decoder in models:
        model_params = list(encoder.parameters())+list(decoder.parameters())
        optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)
        optims.append(optimizer)
    return optims


def get_schedulers(optimizers):
    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               patience=5,
                                                               factor=0.3, verbose=True)
        schedulers.append(scheduler)
    return schedulers


def get_inp_target(encoder, decoder, batch):
    if 'RGB' in encoder.__class__.__name__:
        inp = batch['image'].float()
    elif '2D' in encoder.__class__.__name__:
        inp = batch['pose2d'].float()
    elif '3D' in encoder.__class__.__name__:
        inp = batch['pose3d'].float()

    if 'RGB' in decoder.__class__.__name__:
        target = batch['image'].float()
    elif '2D' in decoder.__class__.__name__:
        target = batch['pose2d'].float()
    elif '3D' in decoder.__class__.__name__:
        target = batch['pose3d'].float()

    return inp, target
