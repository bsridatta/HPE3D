import gc
import os
from collections import OrderedDict, defaultdict

import h5py
import torch
import torch.nn.functional as F

from src.models import KLD, PJPE, reparameterize
from src.processing import post_process, random_rotate, project_3d_to_2d
from src.train_utils import get_inp_target_criterion
from src.viz.mpl_plots import plot_proj, plot_2d, plot_3d

# torch.autograd.set_detect_anomaly(True)

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
        recon_3d = torch.clamp(recon_3d[Ellipsis], min=-1)#, max=2)
        T = torch.tensor((0, 0, 10), device=recon_3d.device, dtype=recon_3d.dtype)

        recon_2d = project_3d_to_2d(recon_3d+T)

        novel_3d_detach = random_rotate(recon_3d.detach())
        novel_3d = random_rotate(recon_3d)

        novel_2d = project_3d_to_2d(novel_3d+T)
        novel_2d_detach = project_3d_to_2d(novel_3d_detach+T)

        ################################################
        # Critic - maximize log(D(x)) + log(1 - D(G(z)))
        ################################################
        critic = model[2].train()
        real_label = 1
        fake_label = 0
        binary_loss = torch.nn.BCELoss()
        critic_optimizer = optimizer[-1]
        critic_optimizer.zero_grad()

        # train with real samples
        labels = torch.full((len(target_2d), 1), real_label,
                            device=config.device, dtype=target_2d.dtype)
        # label smoothing for real labels alone
        label_noise = (torch.rand_like(labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(target_2d)
        critic_loss_real = binary_loss(output, labels)
        critic_loss_real.backward()

        # train with fake samples
        labels.fill_(fake_label)
        # detach to avoid gradient prop to VAE
        output = critic(novel_2d_detach)
        critic_loss_fake = binary_loss(output, labels)
        critic_loss_fake.backward()

        # update critic
        if batch_idx % 1 == 0:
            # Clip grad norm to 1
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
            critic_optimizer.step()

        ################################################
        # Generator - maximize log(D(G(z)))
        ################################################
        # real lables so as to train the vae such that a-
        # -trained discriminator predicts all fake as real

        vae_optimizer.zero_grad()

        labels.fill_(real_label)
        label_noise = (torch.rand_like(labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(novel_2d)

        # Sum 'recon', 'kld' and 'critic' losses
        critic_loss = binary_loss(output, labels)

        recon_loss = criterion(recon_2d, target_2d)
        kld_loss = KLD(mean, logvar, decoder.name)

        loss = config.recon_weight*recon_loss + config.beta*kld_loss + config.critic_weight*critic_loss
        loss.backward()  # Would include VAE and critic but critic not updated

        if False:
            # Clip grad norm to 1
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 2)

            torch.nn.utils.clip_grad_value_(encoder.parameters(), 1000)
            torch.nn.utils.clip_grad_value_(decoder.parameters(), 1000)
            torch.nn.utils.clip_grad_value_(critic.parameters(), 1000)

        # update VAE
        vae_optimizer.step()

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss, "critic_loss": critic_loss,
                "recon_2d": recon_2d, "recon_3d": recon_3d, "novel_2d": novel_2d,
                "target_2d": target_2d, "target_3d": target_3d,
                "critic_loss_real": critic_loss_real, "critic_loss_fake": critic_loss_fake}

        # plot_proj(target[0].detach().cpu(), recon_3d[0].detach().cpu(), recon[0].detach().cpu())

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
        recon_3d = torch.clamp(recon_3d[Ellipsis], min=-1)#, max=2)
        T = torch.tensor((0, 0, 10), device=recon_3d.device, dtype=recon_3d.dtype)

        recon_2d = project_3d_to_2d(recon_3d+T)

        novel_3d_detach = random_rotate(recon_3d.detach())
        novel_3d = random_rotate(recon_3d)

        novel_2d = project_3d_to_2d(novel_3d+T)
        novel_2d_detach = project_3d_to_2d(novel_3d_detach+T)
        ################################################
        # Critic - maximize log(D(x)) + log(1 - D(G(z)))
        ################################################
        critic = model[2].eval()
        real_label = 1
        fake_label = 0
        binary_loss = torch.nn.BCELoss()

        # train with real samples
        labels = torch.full((len(target_2d), 1), real_label,
                            device=config.device, dtype=target_2d.dtype)
        label_noise = (torch.rand_like(labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(target_2d)
        critic_loss_real = binary_loss(output, labels)

        # train with fake samples
        labels.fill_(fake_label)
        # detach to avoid gradient prop to VAE
        output = critic(novel_2d_detach)
        critic_loss_fake = binary_loss(output, labels)

        ################################################
        # Generator - maximize log(D(G(z)))
        ################################################
        # real lables so as to train the vae such that a-
        # -trained discriminator predicts all fake as real

        labels.fill_(real_label)
        label_noise = (torch.rand_like(labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(novel_2d)

        # Sum 'recon', 'kld' and 'critic' losses
        critic_loss = binary_loss(output, labels)
        recon_loss = criterion(recon_2d, target_2d)
        kld_loss = KLD(mean, logvar, decoder.name)

        loss = config.recon_weight*recon_loss + config.beta*kld_loss + config.critic_weight*critic_loss


        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss, "critic_loss": critic_loss,
                "critic_loss_real": critic_loss_real, "critic_loss_fake": critic_loss_fake}

        data = {"recon_2d": recon_2d, "recon_3d": recon_3d, "novel_2d": novel_2d,
                "target_2d": target_2d, "target_3d": target_3d,
                "z": z, "z_attr": batch['action'], "scale_3d": batch['scale_3d']}

    else:
        recon_loss = criterion(recon_3d, target_3d)
        # TODO clip kld loss to prevent explosion
        kld_loss = KLD(mean, logvar, decoder.name)
        loss = recon_loss + config.beta * kld_loss

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss}

        data = {"recon_3d": recon_3d, "target_3d": target_3d,
                "z": z, "z_attr": batch['action']}

    return OrderedDict({'loss': loss, "log": logs,
                        'data': data, "epoch": epoch})


def validation_epoch(config, cb, model, val_loader, epoch, vae_type, normalize_pose=True):
    # note -- model.eval() in validation step
    cb.on_validation_start()

    t_data = defaultdict(list)
    loss_dic = defaultdict(int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device)

            output = _validation_step(batch, batch_idx, model, epoch, config)

            loss_dic['loss'] += output['loss'].item()
            loss_dic['recon_loss'] += output['log']['recon_loss'].item()
            loss_dic['kld_loss'] += output['log']['kld_loss'].item()

            if config.self_supervised:
                loss_dic['critic_loss'] += output['log']['critic_loss'].item()
                loss_dic['critic_loss_real'] += output['log']['critic_loss_real'].item()
                loss_dic['critic_loss_fake'] += output['log']['critic_loss_fake'].item()

            for key in output['data'].keys():
                t_data[key].append(output['data'][key])

            del output
            gc.collect()

    avg_loss = loss_dic['loss']/len(val_loader)  # return for scheduler

    for key in t_data.keys():
        t_data[key] = torch.cat(t_data[key], 0)

    # performance
    if '3D' in model[1].name:
        if normalize_pose and not config.self_supervised:
            t_data['recon_3d'], t_data['target_3d'] = post_process(
                t_data['recon_3d'], t_data['target_3d'])

        elif config.self_supervised:
            t_data['recon_3d'], t_data['target_3d'] = post_process(
                t_data['recon_3d'].to('cpu'), t_data['target_3d'].to('cpu'),
                scale=t_data['scale_3d'].to('cpu'),
                self_supervised=True, procrustes_enabled=False)

        # Speed up procrustes alignment with CPU!
        t_data['recon_3d'].to('cuda')
        t_data['target_3d'].to('cuda')

        pjpe = PJPE(t_data['recon_3d'], t_data['target_3d'])
        avg_pjpe = torch.mean((pjpe), dim=0)
        avg_mpjpe = torch.mean(avg_pjpe).item()
        pjpe = torch.mean(pjpe, dim=1)

        config.logger.log({"pjpe": pjpe.cpu()})

    cb.on_validation_end(config=config, vae_type=vae_type, epoch=epoch, loss_dic=loss_dic,
                         val_loader=val_loader, mpjpe=avg_mpjpe, avg_pjpe=avg_pjpe, pjpe=pjpe, t_data=t_data
                         )

    del loss_dic, t_data
    return avg_loss
