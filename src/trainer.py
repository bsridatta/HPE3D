import gc
import os
from collections import OrderedDict

import h5py
import torch
import torch.nn.functional as F

from src.models import KLD, PJPE, reparameterize
from src.processing import post_process
from src.train_utils import (beta_annealing, beta_cycling,
                             get_inp_target_criterion)


def training_epoch(config, cb, model, train_loader, optimizer, epoch, vae_type):
    """Logic for each epoch"""
    # note -- model.train() in training step

    # TODO perform get_inp_target_criterion for the whole epoch directly
    # or change variantion every batch
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).float()

        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model, config)
        loss = output['loss']
        loss.backward()
        optimizer.step()

        cb.on_train_batch_end(config=config, vae_type=vae_type, epoch=epoch, batch_idx=batch_idx,
                              batch=batch, dataloader=train_loader, output=output, models=model)

    cb.on_train_end(config=config, epoch=epoch)


def validation_epoch(config, cb, model, val_loader, epoch, vae_type, normalize_pose=True):
    # note -- model.eval() in validation step
    
    loss = 0
    recon_loss = 0
    kld_loss = 0

    all_recons = []
    all_targets = []
    all_zs = []
    all_z_attrs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device)

            output = _validation_step(batch, batch_idx, model, epoch, config)

            loss += output['loss'].item()
            recon_loss += output['log']['recon_loss'].item()
            kld_loss += output['log']['kld_loss'].item()

            all_recons.append(output["recon"])
            all_targets.append(output['target'])
            all_zs.append(output["z"])
            all_z_attrs.append(output["z_attr"])

            del output
            gc.collect()

    avg_loss = loss/len(val_loader)  # return for scheduler

    # predictions and targets for performace and visualization
    all_recons = torch.cat(all_recons, 0)
    all_targets = torch.cat(all_targets, 0)
    all_zs = torch.cat(all_zs, 0)
    all_z_attrs = torch.cat(all_z_attrs, 0)

    # performance
    if '3D' in model[1].name:
        if normalize_pose == True:
            all_recons, all_targets = post_process(
                config, all_recons, all_targets)
        pjpe = torch.mean(PJPE(all_recons, all_targets), dim=0)
        mpjpe = torch.mean(pjpe).item()

    cb.on_validation_end(config=config, vae_type=vae_type, epoch=epoch,
                         avg_loss=avg_loss, recon_loss=recon_loss, kld_loss=kld_loss,
                         val_loader=val_loader, mpjpe=mpjpe, pjpe=pjpe,
                         recons=all_recons, targets=all_targets, zs=all_zs, z_attrs=all_z_attrs)

    del loss, kld_loss, recon_loss, all_recons, all_targets, all_zs, all_z_attrs
    return avg_loss


def _training_step(batch, batch_idx, model, config):
    encoder = model[0].train()
    decoder = model[1].train()

    inp, target, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    # clip logvar to prevent inf when exp is calculated
    mean, logvar = encoder(inp)
    logvar = torch.clamp(logvar, max=30)
    z = reparameterize(mean, logvar)
    recon = decoder(z)
    recon = recon.view(target.shape)

    # TODO clip kld loss to prevent explosion
    recon_loss = criterion(recon, target)  # 3D-MSE/MPJPE -- RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar, decoder.__class__.__name__)
    loss = recon_loss + config.beta * kld_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss': loss, 'log': logger_logs})


def _validation_step(batch, batch_idx, model, epoch, config):
    encoder = model[0].eval()
    decoder = model[1].eval()

    inp, target, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar, eval=True)
    recon = decoder(z)
    recon = recon.view(target.shape)

    recon_loss = criterion(recon, target)  # 3D-MPJPE/MSE -- RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar, decoder.__class__.__name__)
    loss = recon_loss + config.beta * kld_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss': loss, "log": logger_logs,
                        "recon": recon, "target": target,
                        "z": z, "z_attr": batch['action'], "epoch": epoch})
