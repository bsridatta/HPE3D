from collections import OrderedDict

import torch
import torch.nn.functional as F

import utils
from models import KLD, MPJPE, reparameterize
from processing import denormalize

beta = 1  # KLD weight


def training_epoch(config, model, train_loader, optimizer, epoch, vae_type):
    # model.train() inside training step

    # TODO perform get_inp_target_criterion for the whole epoch directly
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).float()

        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model)
        _log_training_metrics(config, output, vae_type)
        loss = output['loss_val']
        kld_loss = output['log']['kld_loss']
        recon_loss = output['log']['recon_loss']
        loss.backward()
        optimizer.step()

        # if batch_idx % 100 == 0:
        #     try:
        #         state_dict = model.module.state_dict()
        #     except AttributeError:
        #         state_dict = model.state_dict()

        #     state = {
        #         'model_state_dict': state_dict,
        #         'optimizer_state_dict': optimizer.state_dict()
        #     }
        #     torch.save(state, f'{config.save_dir}/_backup_{config.exp_name}.pt')

        if batch_idx % config.log_interval == 0:
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tReCon: {:.4f}\tKLD: {:.4f}'.format(
                vae_type, epoch, batch_idx *
                len(batch['pose2d']), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                recon_loss.item(), kld_loss.item()))

        del loss, recon_loss, kld_loss


def validation_epoch(config, model, val_loader, epoch, vae_type):
    # model.eval() in validation step
    loss = 0
    acc = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).long()
            output = _validation_step(batch, batch_idx, model, epoch)
            _log_validation_metrics(config, output, vae_type)
            loss += output['loss_val'].item()

    avg_loss = loss/len(val_loader)
    print(f'{vae_type} - Val set: Average Loss: {round(avg_loss,4)}')

    return avg_loss


def _training_step(batch, batch_idx, model):
    encoder = model[0]
    decoder = model[1]
    encoder.train()
    decoder.train()

    inp, target, criterion = utils.get_inp_target_criterion(
        encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    output = decoder(z)

    output = output.view(target.shape)
    recon_loss = criterion(output, target)  # 3D-MPJPE/ RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar)
    loss_val = recon_loss + beta * kld_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss_val': loss_val, 'log': logger_logs})


def _validation_step(batch, batch_idx, model, epoch):
    encoder = model[0]
    decoder = model[1]
    encoder.eval()
    decoder.eval()

    inp, target, criterion = utils.get_inp_target_criterion(
        encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    output = decoder(z)

    output = output.view(target.shape)
    recon_loss = criterion(output, target)
    kld_loss = KLD(mean, logvar)
    loss_val = recon_loss + beta * kld_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss_val': loss_val, "log": logger_logs,  "recon": output,
                        "epoch": epoch})


def evaluate_poses(config, model, val_loader, epoch, vae_type):

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).long()

            # batch['pose2d'] = denormalize(batch)

            output = _validation_step(batch, batch_idx, model, epoch)

            _log_validation_metrics(config, output, vae_type)
            loss += output['loss_val'].item()

    avg_loss = loss/len(val_loader)
    print(f'{vae_type} - Val set: Average Loss: {round(avg_loss,4)}')


def sample_manifold(config, model):
    decoder = model[1]
    decoder.eval()
    with torch.no_grad():
        samples = torch.randn(10, 30).to(config.device)
        samples = decoder(samples)
        if '3D' in decoder.__class__.__name__:
            samples = samples.reshape([-1, 16, 3])
        elif 'RGB' in decoder.__class__.__name__:
            samples = samples.reshape([-1, 256, 256])

        # TODO save as images to tensorboard


def _log_training_metrics(config, output, vae_type):
    config.writer.add_scalars(f"Loss/{vae_type}/Train_Loss", output['log'], 0)
    config.writer.add_scalar(
        f"Total/{vae_type}/Train_Loss", output['loss_val'])


def _log_validation_metrics(config, output, vae_type):
    if output['epoch'] % 2 == 0 and "rgb" in vae_type.split('_')[-1]:
        config.writer.add_image(
            f"Images/{output['epoch']}", output['recon'][0])
    config.writer.add_scalars(f"Loss/{vae_type}/Val_Loss", output['log'], 0)
    config.writer.add_scalar(f"Total/{vae_type}/Val_Loss", output["loss_val"])
