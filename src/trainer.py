from collections import OrderedDict

import torch
import torch.nn.functional as F
import h5py
import gc
import os

import utils
from models import KLD, MPJPE, reparameterize
from processing import denormalize
from viz import plot_diff


def training_epoch(config, model, train_loader, optimizer, epoch, vae_type):

    # TODO perform get_inp_target_criterion for the whole epoch directly
    # or change variantion every batch
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).float()

        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model, config)
        _log_training_metrics(config, output, vae_type)
        loss = output['loss_val']
        kld_loss = output['log']['kld_loss']
        recon_loss = output['log']['recon_loss']
        loss.backward()
        optimizer.step()

        # if batch_idx % 100 == 0:
        # save model

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
    recon_loss = 0
    kld_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).long()
            output = _validation_step(batch, batch_idx, model, epoch, config)
            _log_validation_metrics(config, output, vae_type)
            loss += output['loss_val'].item()

            recon_loss += output['log']['recon_loss'].item()
            kld_loss += output['log']['kld_loss'].item()

    avg_loss = loss/len(val_loader)

    print(f'{vae_type} Validation:',
          f'\t\tLoss: {round(avg_loss,4)}',
          f'\tReCon: {round(recon_loss/len(val_loader), 4)}',
          f'\tKLD: {round(kld_loss/len(val_loader), 4)}')

    del loss, kld_loss, recon_loss
    return avg_loss


def _training_step(batch, batch_idx, model, config):
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
    loss_val = recon_loss + config.beta * kld_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss_val': loss_val, 'log': logger_logs})


def _validation_step(batch, batch_idx, model, epoch, config):
    encoder = model[0]
    decoder = model[1]
    encoder.eval()
    decoder.eval()

    inp, target, criterion = utils.get_inp_target_criterion(
        encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar, eval=True)
    output = decoder(z)

    output = output.view(target.shape)
    recon_loss = criterion(output, target)
    kld_loss = KLD(mean, logvar)
    loss_val = recon_loss + config.beta * kld_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss_val': loss_val, "log": logger_logs,  "recon": output,
                        "epoch": epoch})


def evaluate_poses(config, model, val_loader, epoch, vae_type):
    '''
    Equivalent to merging validation epoch and validation step
    But uses denormalized data to calculate MPJPE
    '''
    norm_stats = h5py.File(
        f"{os.path.dirname(os.path.abspath(__file__))}/data/norm_stats.h5", 'r')

    mpjpes = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).long()

            # get models for eval
            encoder = model[0]
            decoder = model[1]
            encoder.eval()
            decoder.eval()

            inp, target, _ = utils.get_inp_target_criterion(
                encoder, decoder, batch)  # criterion - MSELoss not used

            # forward pass
            mean, logvar = encoder(inp)
            z = reparameterize(mean, logvar, eval=True)
            output = decoder(z)
            output = output.view(target.shape)

            # de-normalize data to original positions
            output = denormalize(
                output,
                torch.tensor(norm_stats['mean3d'], device=config.device),
                torch.tensor(norm_stats['std3d'], device=config.device))
            target = denormalize(
                target,
                torch.tensor(norm_stats['mean3d'], device=config.device),
                torch.tensor(norm_stats['std3d'], device=config.device))

            # since the MPJPE is computed for 17 joints with roots aligned i.e zeroed
            # Not very fair, but the average is with 17 in the denom!
            output = torch.cat(
                (torch.zeros(output.shape[0], 1, 3, device=config.device), output), dim=1)
            target = torch.cat(
                (torch.zeros(target.shape[0], 1, 3, device=config.device), target), dim=1)

            mpjpe = MPJPE(output, target)
            # TODO plot and save samples for say each action to see improvement
            # plot_diff(output[0].numpy(), target[0].numpy(), torch.mean(mpjpe).item())

            mpjpes.append(mpjpe)

    mpjpe = torch.stack(mpjpes, dim=0).sum(dim=0)
    avg_mpjpe = torch.mean(mpjpe).item()

    print(f'{vae_type} - * Mean MPJPE * : {round(avg_mpjpe,4)} \n {mpjpe}')

    del mpjpes
    del mpjpe
    norm_stats.close()
    del norm_stats
    gc.collect()


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
