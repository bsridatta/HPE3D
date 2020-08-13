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

        # optimizer[0].zero_grad()
        output = _training_step(batch, batch_idx, model, config, optimizer)
        # loss = output['loss']
        # loss.backward()
        # optimizer[0].step()

        cb.on_train_batch_end(config=config, vae_type=vae_type, epoch=epoch, batch_idx=batch_idx,
                              batch=batch, dataloader=train_loader, output=output, models=model)

    cb.on_train_end(config=config, epoch=epoch)


def _training_step(batch, batch_idx, model, config, optimizer):
    encoder = model[0].train()
    decoder = model[1].train()

    # len(optimizer) is 1 or 2 with critic optim
    vae_optimizer = optimizer[0]
    inp, target_3d, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    mean, logvar = encoder(inp)
    # clip logvar to prevent inf when exp is calculated
    logvar = torch.clamp(logvar, max=30)
    z = reparameterize(mean, logvar)
    recon_3d = decoder(z)
    recon_3d = recon_3d.view(-1, 16, 3)


    if config.self_supervised:
        # Reprojection
        target_2d = inp
        # TODO recon_3d = torch.clamp(recon_3d, min=10-4, max=10+4)
        recon_3d[:, :, 2] += torch.tensor((10))
        recon_3d_z = (torch.clamp(recon_3d[Ellipsis, -1:], min=1e-12))
        recon_2d = recon_3d[Ellipsis, :-1]/recon_3d_z

        ################################################
        # Critic - maximize log(D(x)) + log(1 - D(G(z)))
        ################################################
        critic = model[2].train()
        real_label = 1
        fake_label = 0
        binary_loss = torch.nn.BCELoss()
        critic_optimizer = optimizer[-1]

        # train with real samples
        critic_optimizer.zero_grad()
        labels = torch.full((len(target_2d), 1), real_label, device=config.device, dtype=target_2d.dtype)
        output = critic(target_2d.detach()) 
        critic_loss_real = binary_loss(output, labels)
        critic_loss_real.backward()

        # train with fake samples
        labels.fill_(fake_label)
        # detach to avoid gradient prop to vae
        output = critic(recon_2d.detach())
        critic_loss_fake = binary_loss(output, labels)
        critic_loss_fake.backward()

        # update critic
        critic_optimizer.step()

        ################################################
        # Generator - maximize log(D(G(z)))
        ################################################
        vae_optimizer.zero_grad()

        # Pass vae output to critic with labels as real
        # TODO rotate and reproject
        labels.fill_(real_label)
        output = critic(recon_2d)
        
        # Sum 'recon', 'kld' and 'critic' losses
        critic_loss = binary_loss(output, labels)
        recon_loss = criterion(recon_2d, target_2d)
        kld_loss = KLD(mean, logvar, decoder.name)
        loss = recon_loss + config.beta * kld_loss + critic_loss

        loss.backward() # Would include vae and critic but critic not updated

        # plot_proj(target[0].detach().cpu(), recon_3d[0].detach().cpu(), recon[0].detach().cpu())
        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss, "critic_loss": critic_loss}

    else:
        vae_optimizer.zero_grad()
        recon_loss = criterion(recon_3d, target_3d)
        # TODO clip kld loss to prevent explosion
        kld_loss = KLD(mean, logvar, decoder.name)
        loss = recon_loss + config.beta * kld_loss
        loss.backward()
        vae_optimizer.step()
    
        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss}

    return OrderedDict({'loss': loss, 'log': logs})


def validation_epoch(config, cb, model, val_loader, epoch, vae_type, normalize_pose=True):
    # note -- model.eval() in validation step

    loss = 0
    recon_loss = 0
    kld_loss = 0
    critic_loss = 0
    t_data = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device)

            output = _validation_step(batch, batch_idx, model, epoch, config)

            loss += output['loss'].item()
            recon_loss += output['log']['recon_loss'].item()
            kld_loss += output['log']['kld_loss'].item()

            if config.self_supervised:
                critic_loss += output['log']['critic_loss'].item()

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
            t_data['recon_3d'], t_data['target_3d'] = post_process(
                config, t_data['recon_3d'], t_data['target_3d'], scale=t_data['scale'])
            pjpe = torch.mean(PJPE(t_data['recon_3d'], t_data['target_3d']), dim=0)

        mpjpe = torch.mean(pjpe).item()

    cb.on_validation_end(config=config, vae_type=vae_type, epoch=epoch, critic_loss=critic_loss,
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
        target = inp.clone()
        # recon = torch.clamp(recon, min=10-4, max=10+4)
        recon[:, :, 2] += torch.tensor((10))
        recon_3d = recon.detach()
        denom = (torch.clamp(recon[Ellipsis, -1:], min=1e-12))
        recon = recon[Ellipsis, :-1]/denom

        # Critic
        critic = model[2].train()
        real_fake = torch.cat([inp, recon], dim=0)
        real_fake_target = torch.cat((
            torch.ones((len(inp), 1), device=config.device),
            torch.zeros((len(recon), 1), device=config.device)
        ), dim=0)
        rand = torch.randint(10, (10, 10), device=config.device)
        rand_idx = torch.randint(0, len(inp)*2-1, size=(len(inp),),
                                 device=config.device)  # .tolist()
        real_fake = real_fake[rand_idx]
        real_fake_target = real_fake_target[rand_idx]

        real_fake_guess = critic(real_fake)
        binary_loss = torch.nn.BCELoss()
        critic_loss = binary_loss(real_fake_guess, real_fake_target)
        critic_weight = 0  # TODO

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

    if config.self_supervised:
        loss = loss + critic_weight * critic_loss
        logs['critic_loss'] = critic_loss

    return OrderedDict({'loss': loss, "log": logs,
                        'data': data, "epoch": epoch})
