import gc
import os
from collections import OrderedDict, defaultdict

import h5py
import torch
import torch.nn.functional as F

from src.models import KLD, PJPE, reparameterize
from src.processing import post_process, project_3d_to_2d
from src.train_utils import get_inp_target_criterion
from src.viz.mpl_plots import plot_proj, plot_2d


def training_epoch(config, cb, model, train_loader, optimizer, epoch, vae_type):
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


def _training_step(batch, batch_idx, model, config):
    encoder = model[0].train()
    decoder = model[1].train()

    inp, target, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    if config.self_supervised:
        target = inp.clone()

    # clip logvar to prevent inf when exp is calculated
    mean, logvar = encoder(inp)
    logvar = torch.clamp(logvar, max=30)
    z = reparameterize(mean, logvar)
    recon = decoder(z)
    recon = recon.view(-1, 16, 3)

    if config.self_supervised:
        # Reprojection
        # recon = torch.clamp(recon, min=10-4, max=10+4)
        recon[:, :, 2] += torch.tensor((10))
        recon_3d = recon.detach()
        denom = (torch.clamp(recon[Ellipsis, -1:], min=1e-12))
        recon = recon[Ellipsis, :-1]/denom

        # CAN REMOVE IT
        # recon = recon[:,:,:-1]
        # recon = project_3d_to_2d(recon, batch)
        # proj_z = recon[:,:,2][:,:,None].repeat(1,1,3)
        # recon = (recon/torch.max(1e-12, proj_z))[:,:,:2]

        # Critic
        # critic = model[2].train()
        # real_fake = recon()
        # critic_loss = guess  # TODO if fake = 1
        # logs['critic_loss'] = critic_loss

    # TODO clip kld loss to prevent explosion
    recon_loss = criterion(recon, target)  # 3D-MSE/MPJPE -- RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar, decoder.name)
    config.beta = 0  # ***************REMOVE***************************
    loss = recon_loss + config.beta * kld_loss
    # plot_proj(target[0].detach().cpu(), recon_3d[0].detach().cpu(), recon[0].detach().cpu())

    logs = {"kld_loss": kld_loss,
            "recon_loss": recon_loss}

    return OrderedDict({'loss': loss, 'log': logs})


def validation_epoch(config, cb, model, val_loader, epoch, vae_type, normalize_pose=True):
    # note -- model.eval() in validation step

    loss = 0
    recon_loss = 0
    kld_loss = 0
    t_data = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device)

            output = _validation_step(batch, batch_idx, model, epoch, config)

            loss += output['loss'].item()
            recon_loss += output['log']['recon_loss'].item()
            kld_loss += output['log']['kld_loss'].item()

            for key in output['data'].keys():
                t_data[key].append(output['data'][key])

            del output
            gc.collect()

    avg_loss = loss/len(val_loader)  # return for scheduler

    for key in t_data.keys():
        t_data[key] = torch.cat(t_data[key], 0)

    # performance
    if '3D' in model[1].name:
        if normalize_pose and not config.self_supervised:
            t_data['recon'], t_data['target'] = post_process(
                config, t_data['recon'], target=t_data['target'])
            pjpe = torch.mean(PJPE(t_data['recon'], t_data['target']), dim=0)

        elif config.self_supervised:
            t_data['recon_3d'], t_data['target_3d'] = post_process(config, t_data['recon_3d'], t_data['target_3d'], scale=t_data['scale'])
            pjpe = torch.mean(PJPE(t_data['recon_3d'], t_data['target_3d']), dim=0)

        mpjpe = torch.mean(pjpe).item()

    cb.on_validation_end(config=config, vae_type=vae_type, epoch=epoch,
                         avg_loss=avg_loss, recon_loss=recon_loss, kld_loss=kld_loss,
                         val_loader=val_loader, mpjpe=mpjpe, pjpe=pjpe, t_data=t_data
                         )

    del loss, kld_loss, recon_loss, t_data
    return avg_loss


def _validation_step(batch, batch_idx, model, epoch, config):
    encoder = model[0].eval()
    decoder = model[1].eval()

    inp, target, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    if config.self_supervised:
        target_3d = target.clone()
        target = inp.clone()

    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar, eval=True)
    recon = decoder(z)
    recon = recon.view(-1, 16, 3)

    if config.self_supervised:
        # Reprojection
        recon[:, :, 2] += torch.tensor((10))
        recon_3d = recon.detach()
        denom = (torch.clamp(recon[Ellipsis, -1:], min=1e-12))
        recon = recon[Ellipsis, :-1]/denom

        data = {"recon": recon, "recon_3d": recon_3d, "input": inp, "target_3d": target_3d,
                "z": z, "z_attr": batch['action'], "scale": batch['scale']}
    else:
        data = {"recon": recon, "target": target, "input": inp,
                "z": z, "z_attr": batch['action']}

    recon_loss = criterion(recon, target)  # 3D-MPJPE/MSE -- RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar, decoder.name)
    loss = recon_loss + config.beta * kld_loss

    logs = {"kld_loss": kld_loss,
            "recon_loss": recon_loss}

    return OrderedDict({'loss': loss, "log": logs,
                        'data': data, "epoch": epoch})
