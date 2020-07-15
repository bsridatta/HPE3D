import sys

import torch

from models import (PJPE, Decoder3D, DecoderRGB, Encoder2D, EncoderRGB)

def max_norm(model, max_val=1, eps=1e-8):
    """clip the norm of the weights to 1, as suggested in Martinez et. al

    Args:
        model (nn.Model): pytorch model
        max_val (int): max norm constraint value. Defaults to 1.
        eps (float): To avoid nan division by zero. Defaults to 1e-8.
    """
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))

def beta_annealing(config, epoch):
    """anneal beta from 0 to 1 during annealing_epochs after waiting for warmup_epochs

    Arguments:
        config {namespace} -- the pipleline configuration
        epoch {integer} -- current training epoch
    """
    # TODO Callback with number of epochs
    if epoch > config.beta_warmup_epochs:
        if epoch <= config.beta_warmup_epochs + config.beta_annealing_epochs:
            config.beta += 0.01/config.beta_annealing_epochs
            print(f"[INFO] Beta increased to: {config.beta}")
        else:
            print(f"[INFO] Beta constant at: {config.beta}")
    else:
        print(f"[INFO] Beta warming: {config.beta}")

    config.logger.log({"beta": config.beta})


def beta_cycling(config, epoch):
    """cycling beta btw 0 and 1 during annealing_epochs after waiting for warmup_epochs

    Arguments:
        config {namespace} -- the pipleline configuration
        epoch {integer} -- current training epoch
    """
    # TODO Callback with number of epochs
    if epoch % config.beta_annealing_epochs == 0:
        config.beta = 0
        print(f"[INFO] Beta reset to: {config.beta}")
    elif epoch % config.beta_annealing_epochs < config.beta_annealing_epochs/2:   
        config.beta += 0.01/config.beta_annealing_epochs*0.5
        print(f"[INFO] Beta increased to: {config.beta}")
    else:
        print(f"[INFO] Beta constant: {config.beta}")

    config.logger.log({"beta": config.beta})





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
        criterion = torch.nn.L1Loss()
        # criterion = PJPE
        # criterion = torch.nn.MSELoss()
    else:
        print("MODEL NOT DEFINED")
        exit()

    return (inp, target, criterion)


def print_pose(pose):
    """print pose with its joint name. for debugging

    Args:
        pose (numpy): 2D or 3D pose
    """
    joint_names = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                   'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    if torch.is_tensor(pose):
        pose = pose.numpy()
    if len(pose) == 17:
        for x in range(len(pose)):
            print(f'{joint_names[x]:10} {pose[x]}')
    else:
        for x in range(len(pose)):
            print(f'{joint_names[x+1]:10} {pose[x]}')
