import gc
import os
from collections import OrderedDict, defaultdict

import h5py
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from src.models import KLD, PJPE, reparameterize
from src.processing import post_process, random_rotate, project_3d_to_2d
from src.train_utils import get_inp_target_criterion
from src.viz.mpl_plots import plot_proj, plot_2d, plot_3d

# torch.autograd.set_detect_anomaly(True)


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
        ################################################
        # Get novel training data from reprojection
        ################################################
        target_2d = inp.detach()
        
        # enforce unit recon if above root is scaled to 1
        recon_3d = recon_3d*1.3
        T = torch.tensor((0, 0, 10), device=recon_3d.device, dtype=recon_3d.dtype)

        recon_2d = project_3d_to_2d(recon_3d+T)

        novel_3d_detach = random_rotate(recon_3d.detach())
        novel_3d = random_rotate(recon_3d)

        novel_2d = project_3d_to_2d(novel_3d+T)
        novel_2d_detach = project_3d_to_2d(novel_3d_detach+T)

        # uncomment to use the same fake for training critic and the generator
        # novel_2d_detach = novel_2d.detach()

        ################################################
        # Critic
        ################################################
        critic = model[2].train()
        critic_optimizer = optimizer[-1]
        critic_optimizer.zero_grad()

        # confuse critic ??#############################

        # train on real samples
        output_real = critic(target_2d)
        D_x = output_real.mean().item()

        # train on fake samples
        output_fake = critic(novel_2d_detach)  # detach to avoid gradient prop to VAE
        D_G_z1 = output_fake.mean().item()

        # gp
        lambd_gp = 10
        gradient_penalty = compute_gradient_penalty(critic, target_2d.data, novel_2d_detach.data)
        
        # critic loss
        critic_loss = -1*torch.mean(output_real) + torch.mean(output_fake) +\
            lambd_gp * gradient_penalty

        # update critic
        if batch_idx % 1 == 0:
            critic_loss.backward()
            critic_optimizer.step()

        ################################################
        # Generator
        ################################################
        vae_optimizer.zero_grad()

        output_gen = critic(novel_2d)
        D_G_z2 = output_gen.mean().item()

        gen_loss = -1*torch.mean(output_gen)

        recon_loss = criterion(recon_2d, target_2d)
        kld_loss = KLD(mean, logvar, decoder.name)

        lambd_recon = 1
        lambd_critic = config.critic_weight

        loss = lambd_recon * recon_loss + config.beta * kld_loss + lambd_critic * gen_loss
        loss *= 10  # standardize for comparision with other loss variants

        loss.backward()  # Would include VAE and critic but critic not updated


        if False:
            # Clip grad norm to 1 *****************************************
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 2)

            torch.nn.utils.clip_grad_value_(encoder.parameters(), 1000)
            torch.nn.utils.clip_grad_value_(decoder.parameters(), 1000)
            torch.nn.utils.clip_grad_value_(critic.parameters(), 1000)

        # update VAE
        if batch_idx % 1 == 0:
            vae_optimizer.step()

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss, "gen_loss": gen_loss,
                "critic_loss": critic_loss, "recon_2d": recon_2d, "recon_3d": recon_3d,
                "novel_2d": novel_2d, "target_2d": target_2d, "target_3d": target_3d,
                "D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2}

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
        ################################################
        # Get novel training data from reprojection
        ################################################        
        target_2d = inp.detach()

        # enforce unit recon if above root is scaled to 1
        recon_3d = recon_3d*1.3
        T = torch.tensor((0, 0, 10), device=recon_3d.device, dtype=recon_3d.dtype)

        recon_2d = project_3d_to_2d(recon_3d+T)

        novel_3d_detach = random_rotate(recon_3d.detach())
        novel_3d = random_rotate(recon_3d)

        novel_2d = project_3d_to_2d(novel_3d+T)
        novel_2d_detach = project_3d_to_2d(novel_3d_detach+T)

        # uncomment to use the same fake for training critic and the generator
        # novel_2d_detach = novel_2d.detach()

        ################################################
        # Critic
        ################################################
        critic = model[2].eval()

        # train on real samples
        output_real = critic(target_2d)
        D_x = output_real.mean().item()

        # train on fake samples
        output_fake = critic(novel_2d_detach) # detach to avoid gradient prop to VAE
        D_G_z1 = output_fake.mean().item()


        # gp
        lambd_gp = 10
        # gradient_penalty = compute_gradient_penalty(critic, target_2d.data, novel_2d_detach.data)
        
        # critic loss
        critic_loss = -1*torch.mean(output_real) + torch.mean(output_fake) #+\
            # lambd_gp * gradient_penalty

        ################################################
        # Generator
        ################################################
        output_gen = critic(novel_2d)
        D_G_z2 = output_gen.mean().item()

        gen_loss = -1*torch.mean(output_gen)

        recon_loss = criterion(recon_2d, target_2d)
        kld_loss = KLD(mean, logvar, decoder.name)

        lambd_recon = 1
        lambd_critic = config.critic_weight

        loss = lambd_recon * recon_loss + config.beta * kld_loss + lambd_critic * gen_loss
        loss *= 10  # standardize for comparision with other loss variants

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss,
                "gen_loss": gen_loss, "critic_loss": critic_loss,
                "D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2}

        data = {"recon_2d": recon_2d, "recon_3d": recon_3d, "novel_2d": novel_2d,
                "target_2d": target_2d, "target_3d": target_3d,
                "z": z, "action": batch['action']}

    else:
        recon_loss = criterion(recon_3d, target_3d)
        # TODO clip kld loss to prevent explosion
        kld_loss = KLD(mean, logvar, decoder.name)
        loss = recon_loss + config.beta * kld_loss

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss}

        data = {"recon_3d": recon_3d, "target_3d": target_3d,
                "z": z, "action": batch['action']}

    return OrderedDict({'loss': loss, "log": logs,
                        'data': data, "epoch": epoch})


def training_epoch(config, cb, model, train_loader, optimizer, epoch, vae_type):
    # note -- model.train() in training step
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).float()

        output = _training_step(batch, batch_idx, model, config, optimizer)

        cb.on_train_batch_end(config=config, vae_type=vae_type, epoch=epoch, batch_idx=batch_idx,
                              batch=batch, dataloader=train_loader, output=output, models=model)
    cb.on_train_end(config=config, epoch=epoch)


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
                loss_dic['gen_loss'] += output['log']['gen_loss'].item()
                loss_dic['critic_loss'] += output['log']['critic_loss'].item()
                loss_dic['D_x'] += output['log']['D_x']
                loss_dic['D_G_z1'] += output['log']['D_G_z1']
                loss_dic['D_G_z2'] += output['log']['D_G_z2']

            for key in output['data'].keys():
                t_data[key].append(output['data'][key])

            del output
            gc.collect()

    avg_loss = loss_dic['loss']/len(val_loader)  # return for scheduler

    for key in t_data.keys():
        t_data[key] = torch.cat(t_data[key], 0)

    # performance
    t_data['recon_3d_org'] = t_data['recon_3d'].detach()
    if '3D' in model[1].name:
        if normalize_pose and not config.self_supervised:
            t_data['recon_3d'], t_data['target_3d'] = post_process(
                t_data['recon_3d'], t_data['target_3d'])

        elif config.self_supervised:
            t_data['recon_3d'], t_data['target_3d'] = post_process(
                t_data['recon_3d'].to('cpu'), t_data['target_3d'].to('cpu'),
                # scale=t_data['scale_3d'].to('cpu'),
                self_supervised=True, procrustes_enabled=True)

        # Speed up procrustes alignment with CPU!
        t_data['recon_3d'].to('cuda')
        t_data['target_3d'].to('cuda')

        pjpe_ = PJPE(t_data['recon_3d'], t_data['target_3d'])  # per sample per joint [n,j]
        avg_pjpe = torch.mean((pjpe_), dim=0)  # across all samples per joint [j]
        avg_mpjpe = torch.mean(avg_pjpe).item()  # across all samples all joint [1]
        pjpe = torch.mean(pjpe_, dim=1)  # per sample all joints [n]

        actions = t_data['action']
        mpjpe_pa = {}  # per action
        for i in torch.unique(actions):
            res = torch.mean(pjpe[actions == i])
            mpjpe_pa[i.item()] = res.item()

        config.logger.log({"pjpe": pjpe.cpu()})

    cb.on_validation_end(config=config, vae_type=vae_type, epoch=epoch, loss_dic=loss_dic, mpjpe_pa=mpjpe_pa,
                         val_loader=val_loader, mpjpe=avg_mpjpe, avg_pjpe=avg_pjpe, pjpe=pjpe, t_data=t_data
                         )

    del loss_dic, t_data
    return avg_loss


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP
    Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py"""
    Tensor = torch.cuda.FloatTensor if real_samples.device.type=="cuda" else torch.FloatTensor 
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = autograd.Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
