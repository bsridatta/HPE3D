import gc
from collections import OrderedDict, defaultdict

import torch
from torch import nn
from src.models import KLD, PJPE, reparameterize
from src.processing import post_process, random_rotate, project_3d_to_2d
from src.train_utils import get_inp_target_criterion

# torch.autograd.set_detect_anomaly(True)


def _training_step(batch, batch_idx, model, config, optimizer):
    encoder = model[0].train()
    decoder = model[1].train()

    # len(optimizer) is 1 or 2 with critic optim
    vae_optimizer = optimizer[0]

    inp, target_3d, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    if config.p_miss:

        pose = inp
        pose2d_org = inp.clone()

        # index of poses to be incomplete
        incomplete_poses_ids = torch.multinomial(torch.ones(
            pose.shape[0]), int(pose.shape[0]*config.p_miss), replacement=False)

        # probablity to choose a joint to miss
        p_limbs = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        p_limbs = torch.Tensor(p_limbs).to(pose.device)
        p_limbs = p_limbs.repeat(len(incomplete_poses_ids), 1)

        # 2 random joints to exclude for each missing pose
        # make 0.5 of them to miss 1 joint only by duplicting the joint id
        rand_joints = torch.multinomial(p_limbs, 2, replacement=False)
        rand_joints[:rand_joints.shape[0]//2][:,
                                              1] = rand_joints[:rand_joints.shape[0]//2][:, 0]

        # repeat incomplete pose ids for vectorization
        incomplete_poses_ids = incomplete_poses_ids.view(-1, 1).repeat(1, 2)

        # zero the random joints of the 'tobe' incomplete poses
        pose[incomplete_poses_ids, rand_joints, :] = 0

        inp = pose  # not needed

    mean, logvar = encoder(inp)

    # clip logvar to prevent inf when exp is calculated
    logvar = torch.clamp(logvar, max=30)
    z = reparameterize(mean, logvar)
    recon_3d = decoder(z)
    recon_3d = recon_3d.view(-1, model[0].n_joints, 3)

    if config.self_supervised:

        # Reprojection
        target_2d = inp.detach().clone()
        noised_real = add_noise(inp.detach().clone(), config.noise_level)

        # enforce unit recon if above root is scaled to 1
        recon_3d = recon_3d*1.3

        T = torch.tensor((0, 0, 10), device=recon_3d.device,
                         dtype=recon_3d.dtype)

        recon_2d = project_3d_to_2d(recon_3d+T)

        novel_3d_detach = random_rotate(recon_3d.detach())
        novel_3d = random_rotate(recon_3d)

        novel_2d = project_3d_to_2d(novel_3d+T)
        novel_2d_detach = project_3d_to_2d(novel_3d_detach+T)

        # Use the same fake for training critic and the generator
        novel_2d_detach = novel_2d.detach()

        ################################################
        # Critic - maximize log(D(x)) + log(1 - D(G(z)))
        ################################################
        critic = model[2].train()
        real_label = 1
        fake_label = 0
        binary_loss = nn.BCELoss()
        critic_optimizer = optimizer[-1]
        critic_optimizer.zero_grad(set_to_none=True)

        # confuse critic
        # if batch_idx % 7 == 0:
        #     real_label = 0
        #     fake_label = 1

        # train with real samples
        labels = torch.full((len(target_2d), 1), real_label,
                            device=config.device, dtype=target_2d.dtype)
        # label smoothing for real labels alone
        label_noise = (torch.rand_like(
            labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(noised_real)
        critic_loss_real = binary_loss(output, labels)
        critic_loss_real.backward()
        D_x = output.mean().item()

        # train with fake samples
        labels.fill_(fake_label)
        # label smoothing for real labels alone *** not TODO
        # label_noise = (torch.rand_like(labels, device=labels.device)*(0.0-0.3)) + 0.3
        # labels = labels * label_noise

        # detach to avoid gradient prop to VAE
        output = critic(novel_2d_detach)
        critic_loss_fake = binary_loss(output, labels)
        critic_loss_fake.backward()
        D_G_z1 = output.mean().item()

        critic_loss = critic_loss_real+critic_loss_fake

        # update critic
        if batch_idx % 1 == 0:
            # Clip grad norm to 1 ********************************
            nn.utils.clip_grad_norm_(critic.parameters(), 1)
            critic_optimizer.step()

        ################################################
        # Generator - maximize log(D(G(z)))
        ################################################
        # real lables so as to train the vae such that a-
        # -trained discriminator predicts all fake as real

        # required if the real and fake are flipped in critic training
        real_label = 1
        fake_label = 0

        vae_optimizer.zero_grad(set_to_none=True)

        labels.fill_(real_label)
        label_noise = (torch.rand_like(
            labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(novel_2d)

        # Sum 'recon', 'kld' and 'critic' losses
        gen_loss = binary_loss(output, labels)

        if config.p_miss:
            recon_2d_org = recon_2d.clone().detach()
            recon_2d[incomplete_poses_ids, rand_joints, :] = 0
            recon_loss = criterion(recon_2d, target_2d)
            recon_2d = recon_2d_org
        else:
            recon_loss = criterion(recon_2d, target_2d)

        kld_loss = KLD(mean, logvar, decoder.name)

        # lambda_kld is used to compute the beta coeff
        loss = config.lambda_gen*recon_loss + config.beta * kld_loss + \
            config.lambda_disc*gen_loss

        loss *= 10

        loss.backward()  # Would include VAE and critic but critic not updated

        D_G_z2 = output.mean().item()

        if True:
            # Clip grad norm to 1 *****************************************
            nn.utils.clip_grad_norm_(encoder.parameters(), 2)
            nn.utils.clip_grad_norm_(decoder.parameters(), 2)
            nn.utils.clip_grad_norm_(critic.parameters(), 2)

            nn.utils.clip_grad_value_(encoder.parameters(), 1000)
            nn.utils.clip_grad_value_(decoder.parameters(), 1000)
            nn.utils.clip_grad_value_(critic.parameters(), 1000)

        # update VAE
        if batch_idx % 1 == 0:
            vae_optimizer.step()

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss, "gen_loss": gen_loss,
                "critic_loss": critic_loss, "recon_2d": recon_2d, "recon_3d": recon_3d,
                "novel_2d": novel_2d, "target_2d": target_2d, "target_3d": target_3d,
                "D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2}

    else:
        vae_optimizer.zero_grad(set_to_none=True)
        recon_loss = criterion(recon_3d, target_3d)
        # TODO clip kld loss to prevent explosion
        kld_loss = KLD(mean, logvar, decoder.name)
        loss = recon_loss + config.beta * kld_loss
        loss.backward()
        vae_optimizer.step()

        logs = {"kld_loss": kld_loss, "recon_loss": recon_loss}

    return OrderedDict({'loss': loss, 'log': logs})


def _validation_step(batch, batch_idx, model, epoch, config, eval=True):
    encoder = model[0].eval()
    decoder = model[1].eval()

    inp, target_3d, criterion = get_inp_target_criterion(
        encoder, decoder, batch)

    if config.p_miss:

        pose = inp
        pose2d_org = inp.clone()

        # index of poses to be incomplete
        incomplete_poses_ids = torch.multinomial(torch.ones(
            pose.shape[0]), int(pose.shape[0]*config.p_miss), replacement=False)

        # probablity to choose a joint to miss
        p_limbs = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        p_limbs = torch.Tensor(p_limbs).to(pose.device)
        p_limbs = p_limbs.repeat(len(incomplete_poses_ids), 1)

        # 2 random joints to exclude for each missing pose
        # make 0.5 of them to miss 1 joint only by duplicting the joint id
        rand_joints = torch.multinomial(p_limbs, 2, replacement=False)
        rand_joints[:rand_joints.shape[0]//2][:,
                                              1] = rand_joints[:rand_joints.shape[0]//2][:, 0]

        # repeat incomplete pose ids for vectorization
        incomplete_poses_ids = incomplete_poses_ids.view(-1, 1).repeat(1, 2)

        # zero the random joints of the 'tobe' incomplete poses
        pose[incomplete_poses_ids, rand_joints, :] = 0

        inp = pose  # not needed

    mean, logvar = encoder(inp)
    # clip logvar to prevent inf when exp is calculated
    logvar = torch.clamp(logvar, max=30)
    z = reparameterize(mean, logvar, eval)
    recon_3d = decoder(z)
    recon_3d = recon_3d.view(-1, model[0].n_joints, 3)

    if config.self_supervised:
        # criterion = nn.MSELoss()

        # Reprojection
        target_2d = inp.detach()
        noised_real = add_noise(inp.detach(), config.noise_level)
        # enforce unit recon if above root is scaled to 1
        recon_3d = recon_3d*1.3

        T = torch.tensor((0, 0, 10), device=recon_3d.device,
                         dtype=recon_3d.dtype)

        recon_2d = project_3d_to_2d(recon_3d+T)

        novel_3d_detach = random_rotate(recon_3d.detach())
        novel_3d = random_rotate(recon_3d)

        novel_2d = project_3d_to_2d(novel_3d+T)
        novel_2d_detach = project_3d_to_2d(novel_3d_detach+T)

        # Use the same fake for training critic and the generator
        novel_2d_detach = novel_2d.detach()

        ################################################
        # Critic - maximize log(D(x)) + log(1 - D(G(z)))
        ################################################
        critic = model[2].eval()
        real_label = 1
        fake_label = 0
        binary_loss = nn.BCELoss()

        # validate on real samples
        labels = torch.full((len(target_2d), 1), real_label,
                            device=recon_3d.device, dtype=target_2d.dtype)
        label_noise = (torch.rand_like(
            labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(noised_real)
        critic_loss_real = binary_loss(output, labels)
        D_x = output.mean().item()

        # validate on fake samples
        labels.fill_(fake_label)
        # label smoothing for real labels alone *** not TODO
        # label_noise = (torch.rand_like(labels, device=labels.device)*(0.0-0.3)) + 0.3
        # labels = labels * label_noise

        # detach to avoid gradient prop to VAE
        output = critic(novel_2d_detach)
        critic_loss_fake = binary_loss(output, labels)
        D_G_z1 = output.mean().item()

        critic_loss = critic_loss_real+critic_loss_fake
        ################################################
        # Generator - maximize log(D(G(z)))
        ################################################
        # real lables so as to train the vae such that a-
        # -trained discriminator predicts all fake as real

        # required if the real and fake are flipped in critic training
        real_label = 1
        fake_label = 0

        labels.fill_(real_label)
        label_noise = (torch.rand_like(
            labels, device=labels.device)*(0.7-1.2)) + 1.2
        labels = labels * label_noise

        output = critic(novel_2d)

        # Sum 'recon', 'kld' and 'critic' losses
        gen_loss = binary_loss(output, labels)

        if config.p_miss:
            recon_2d_org = recon_2d.clone().detach()
            recon_2d[incomplete_poses_ids, rand_joints, :] = 0
            recon_loss = criterion(recon_2d, target_2d)
            recon_2d = recon_2d_org
        else:
            recon_loss = criterion(recon_2d, target_2d)

        kld_loss = KLD(mean, logvar, decoder.name)

        # lambda_kld is used to compute the beta coeff
        loss = config.lambda_gen*recon_loss + config.beta * kld_loss + \
            config.lambda_disc*gen_loss
        loss *= 10

        D_G_z2 = output.mean().item()

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
        t_data['recon_3d'].to(config.device)
        t_data['target_3d'].to(config.device)

        # per sample per joint [n,j]
        pjpe_ = PJPE(t_data['recon_3d'], t_data['target_3d'])
        # across all samples per joint [j]
        avg_pjpe = torch.mean((pjpe_), dim=0)
        # across all samples all joint [1]
        avg_mpjpe = torch.mean(avg_pjpe).item()
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


def add_noise(pose, noise_level):
    noise = torch.randn(pose.shape).to(pose.device) * (pose*noise_level)
    pose = pose+noise

    return pose
