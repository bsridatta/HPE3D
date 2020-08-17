import gc
import os
from collections import OrderedDict, defaultdict

import h5py
import torch
import torch.nn.functional as F

from src.models import KLD, PJPE, reparameterize
from src.processing import post_process, random_rotate_and_project_3d_to_2d
from src.train_utils import get_inp_target_criterion
from src.viz.mpl_plots import plot_proj, plot_2d


def training_epoch(config, cb, model, train_loader, optimizer, epoch, vae_type):
    # note -- model.train() in training step

    # TODO perform get_inp_target_criterion for the whole epoch directly
    # or change variantion every batch
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).float()

        output = _training_step(batch, batch_idx, model, config, optimizer)

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
        target_2d = inp.detach()
        # ???????????????????
        # recon_3d = torch.clamp(recon_3d, min=10-10, max=10+10)

        recon_3d[:, :, 2] += torch.tensor((10))
        recon_2d = random_rotate_and_project_3d_to_2d(recon_3d, random_rotate=False)
        novel_2d_view = random_rotate_and_project_3d_to_2d(recon_3d.detach())

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
        labels = torch.full((len(target_2d), 1), real_label,
                            device=config.device, dtype=target_2d.dtype)
        output = critic(target_2d.detach())
        critic_loss_real = binary_loss(output, labels)
        critic_loss_real.backward()

        # train with fake samples
        labels.fill_(fake_label)
        # detach to avoid gradient prop to VAE
        output = critic(novel_2d_view.detach())
        critic_loss_fake = binary_loss(output, labels)
        critic_loss_fake.backward()

        # update critic
        critic_optimizer.step()

        ################################################
        # Generator - maximize log(D(G(z)))
        ################################################
        vae_optimizer.zero_grad()

        # real lables so as to train the vae such that a-
        # -trained discriminator predicts all fake as real
        labels.fill_(real_label)
        output = critic(novel_2d_view)

        # Sum 'recon', 'kld' and 'critic' losses
        critic_loss = binary_loss(output, labels)
        recon_loss = criterion(recon_2d, target_2d)
        kld_loss = KLD(mean, logvar, decoder.name)
        critic_weight = 1
        loss = recon_loss + config.beta*kld_loss + critic_loss * critic_weight
        loss.backward()  # Would include VAE and critic but critic not updated

        # update VAE
        vae_optimizer.step()
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
            # TODO REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!?????????????????????????
            break
    avg_loss = loss/len(val_loader)  # return for scheduler

    for key in t_data.keys():
        t_data[key] = torch.cat(t_data[key], 0)

    # performance
    if '3D' in model[1].name:
        if normalize_pose and not config.self_supervised:
            t_data['recon_3d'], t_data['target_3d'] = post_process(
                config, t_data['recon_3d'], target=t_data['target_3d'])
            pjpe = torch.mean(PJPE(t_data['recon_3d'], t_data['target_3d']), dim=0)

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

    inp, target_3d, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    mean, logvar = encoder(inp)
    # clip logvar to prevent inf when exp is calculated
    logvar = torch.clamp(logvar, max=30)
    z = reparameterize(mean, logvar, eval=True)
    recon_3d = decoder(z)
    recon_3d = recon_3d.view(-1, 16, 3)

    if config.self_supervised:
        # Reprojection
        target_2d = inp.detach()
        # ???????????????????
        # recon_3d = torch.clamp(recon_3d, min=10-10, max=10+10)

        recon_3d[:, :, 2] += torch.tensor((10))
        recon_2d = random_rotate_and_project_3d_to_2d(recon_3d, random_rotate=False)
        novel_2d_view = random_rotate_and_project_3d_to_2d(recon_3d.detach())

        ################################################
        # Critic
        ################################################
        critic = model[2].eval()
        real_label = 1
        fake_label = 0
        binary_loss = torch.nn.BCELoss()

        # eval with real samples
        labels = torch.full((len(target_2d), 1), real_label,
                            device=config.device, dtype=target_2d.dtype)
        output = critic(target_2d.detach())
        critic_loss_real = binary_loss(output, labels)

        # eval with fake samples
        labels.fill_(fake_label)
        output = critic(novel_2d_view.detach())
        critic_loss_fake = binary_loss(output, labels)

        ################################################
        # Generator
        ################################################
        # real lables so as to train the vae such that a
        # trained discriminator predicts all fake as real
        labels.fill_(real_label)
        output = critic(novel_2d_view)

        # Sum 'recon', 'kld' and 'critic' losses
        critic_loss = binary_loss(output, labels)
        recon_loss = criterion(recon_2d, target_2d)
        kld_loss = KLD(mean, logvar, decoder.name)
        critic_weight = 1
        loss = recon_loss + config.beta*kld_loss + critic_loss * critic_weight

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss, "critic_loss": critic_loss}

        data = {"recon_2d": recon_2d, "recon_3d": recon_3d, "input": inp, "target_3d": target_3d,
                "z": z, "z_attr": batch['action'], "scale": batch['scale']}

    else:
        recon_loss = criterion(recon_3d, target_3d)
        # TODO clip kld loss to prevent explosion
        kld_loss = KLD(mean, logvar, decoder.name)
        loss = recon_loss + config.beta * kld_loss

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss}

        data = {"recon": recon, "target": target, "input": inp,
                "z": z, "z_attr": batch['action']}


    return OrderedDict({'loss': loss, "log": logs,
                        'data': data, "epoch": epoch})
