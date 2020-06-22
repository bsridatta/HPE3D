import gc
import os
from collections import OrderedDict

import h5py
import torch
import torch.nn.functional as F

from models import KLD, PJPE, reparameterize
from processing import post_process
from utils import get_inp_target_criterion, beta_annealing


def training_epoch(config, model, train_loader, optimizer, epoch, vae_type):
    # note -- model.train() in training step

    # TODO perform get_inp_target_criterion for the whole epoch directly
    # or change variantion every batch
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).float()

        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model, config)
        _log_training_metrics(config, output, vae_type)

        kld_loss = output['log']['kld_loss']
        recon_loss = output['log']['recon_loss']
        loss = output['loss']

        loss.backward()
        optimizer.step()

        # if batch_idx % 100 == 0:
        # backup model?

        print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tReCon: {:.4f}\tKLD: {:.4f}'.format(
            vae_type, epoch, batch_idx * len(batch['pose2d']),
            len(train_loader.dataset), 100. *
            batch_idx / len(train_loader),
            loss.item(), recon_loss.item(), kld_loss.item()))

    # Anneal beta 0 - 1
    beta_annealing(config, epoch)

    del loss, recon_loss, kld_loss, output


def validation_epoch(config, model, val_loader, epoch, vae_type, normalize_pose=True):
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

            all_recons.append(output["recon"])
            all_targets.append(output['target'])
            all_zs.append(output["z"])
            all_z_attrs.append(output["z_attr"])

            loss += output['loss'].item()
            recon_loss += output['log']['recon_loss'].item()
            kld_loss += output['log']['kld_loss'].item()
            del output
            gc.collect()

    avg_loss = loss/len(val_loader)  # return for scheduler

    # Return Predictions and Target for performace visualization
    all_recons = torch.cat(all_recons, 0)
    all_targets = torch.cat(all_targets, 0)
    all_zs = torch.cat(all_zs, 0)
    all_z_attrs = torch.cat(all_z_attrs, 0)

    if '3D' in model[1].name and normalize_pose == True:
        all_recons, all_targets = post_process(config, all_recons, all_targets)

    # Logging
    print(f'{vae_type} Validation:',
          f'\t\tLoss: {round(avg_loss,4)}',
          f'\tReCon: {round(recon_loss/len(val_loader), 4)}',
          f'\tKLD: {round(kld_loss/len(val_loader), 4)}')

    avg_output = {}
    avg_output['loss'] = avg_loss
    avg_output['log'] = {}
    avg_output['log']['recon_loss'] = recon_loss/len(val_loader)
    avg_output['log']['kld_loss'] = kld_loss/len(val_loader)

    _log_validation_metrics(config, avg_output, vae_type)

    del loss, kld_loss, recon_loss, avg_output

    return avg_loss, all_recons, all_targets, all_zs, all_z_attrs


def _training_step(batch, batch_idx, model, config):
    encoder = model[0]
    decoder = model[1]
    encoder.train()
    decoder.train()

    inp, target, criterion = get_inp_target_criterion(
        encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    recon = decoder(z)
    recon = recon.view(target.shape)

    recon_loss = criterion(recon, target)  # 3D-MSE/MPJPE -- RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar, decoder.__class__.__name__)
    loss = recon_loss + config.beta * kld_loss
    # TODO clip kld loss to prevent explosion

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss': loss, 'log': logger_logs})


def _validation_step(batch, batch_idx, model, epoch, config):
    encoder = model[0]
    decoder = model[1]
    encoder.eval()
    decoder.eval()

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

    # TODO could remove epoch
    return OrderedDict({'loss': loss, "log": logger_logs,
                        "recon": recon, "target": target,
                        "z": z, "z_attr": batch['action'], "epoch": epoch})


def _log_training_metrics(config, output, vae_type):
    config.logger.log({
        f"{vae_type}": {
            "train": {
                "kld_loss": output['log']['kld_loss'],
                "recon_loss": output['log']['recon_loss'],
                "total_train": output['loss']
            }
        }
    })


def _log_validation_metrics(config, output, vae_type):
    # TODO can have this in eval instead and skip logging val
    # if output['epoch'] % 2 == 0 and "rgb" in vae_type.split('_')[-1]:
    #     # TODO change code to wandb
    #     config.logger.log(
    #         f"Images/{output['epoch']}", output['recon'][0])

    config.logger.log({
        f"{vae_type}": {
            "val": {
                "kld_loss": output['log']['kld_loss'],
                "recon_loss": output['log']['recon_loss'],
                "total_val": output['loss']
            }
        }
    })
