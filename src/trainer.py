import gc
import os
from collections import OrderedDict

import h5py
import torch
import torch.nn.functional as F

from models import KLD, MPJPE, reparameterize
from processing import denormalize
from utils import get_inp_target_criterion
from viz import plot_diff, plot_diffs


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
        # save model

        print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tReCon: {:.4f}\tKLD: {:.4f}'.format(
            vae_type, epoch, batch_idx * len(batch['pose2d']),
            len(train_loader.dataset), 100. *
            batch_idx / len(train_loader),
            loss.item(), recon_loss.item(), kld_loss.item()))

        del loss, recon_loss, kld_loss


def validation_epoch(config, model, val_loader, epoch, vae_type):
    # note -- model.eval() in validation step
    loss = 0
    recon_loss = 0
    kld_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device)

            output = _validation_step(batch, batch_idx, model, epoch, config)
            # logging for val steps..skews the plot as wandb steps increase
            # _log_validation_metrics(config, output, vae_type)

            loss += output['loss'].item()
            recon_loss += output['log']['recon_loss'].item()
            kld_loss += output['log']['kld_loss'].item()

    avg_loss = loss/len(val_loader)  # return for scheduler

    print(f'{vae_type} Validation:',
          f'\t\tLoss: {round(avg_loss,4)}',
          f'\tReCon: {round(recon_loss/len(val_loader), 4)}',
          f'\tKLD: {round(kld_loss/len(val_loader), 4)}')

    # use _log_validation_metrics per epoch rather than batch
    avg_output = {}
    avg_output['loss'] = avg_loss
    avg_output['log'] = {}
    avg_output['log']['recon_loss'] = recon_loss/len(val_loader)
    avg_output['log']['kld_loss'] = kld_loss/len(val_loader)

    _log_validation_metrics(config, avg_output, vae_type)

    del loss, kld_loss, recon_loss

    return avg_loss


def _training_step(batch, batch_idx, model, config):
    encoder = model[0]
    decoder = model[1]
    encoder.train()
    decoder.train()

    inp, target, criterion = get_inp_target_criterion(
        encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    output = decoder(z)
    output = output.view(target.shape)

    recon_loss = criterion(output, target)  # 3D-MSE/MPJPE -- RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar, decoder.__class__.__name__)
    loss = recon_loss + config.beta * kld_loss

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
    output = decoder(z)
    output = output.view(target.shape)

    recon_loss = criterion(output, target)  # 3D-MPJPE/MSE -- RGB/2D-L1/BCE
    kld_loss = KLD(mean, logvar, decoder.__class__.__name__)
    loss = recon_loss + config.beta * kld_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    # TODO could remove epoch and recon
    return OrderedDict({'loss': loss, "log": logger_logs,  "recon": output,
                        "epoch": epoch})


def evaluate_poses(config, model, val_loader, epoch, vae_type):
    '''
    Equivalent to merging validation epoch and validation step
    But uses denormalized data to calculate MPJPE
    '''
    ann_file_name = config.annotation_file.split('/')[-1].split('.')[0]
    norm_stats = h5py.File(
        f"{os.path.dirname(os.path.abspath(__file__))}/data/norm_stats_{ann_file_name}_911.h5", 'r')

    pjpes = []

    zs = []
    actions = []

    outputs = []
    targets = []
    errors = []

    n_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).long()

            # get models for eval
            encoder = model[0]
            decoder = model[1]
            encoder.eval()
            decoder.eval()

            inp, target, _ = get_inp_target_criterion(
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

            pjpe = MPJPE(output, target)
            # TODO plot and save samples for say each action to see improvement
            # plot_diff(output[0].numpy(), target[0].numpy(), torch.mean(mpjpe).item())
            pjpes.append(torch.sum(pjpe, dim=0))
            n_samples += pjpe.shape[0]  # to calc overall mean

            # Poses Viz
            if batch_idx < 1:
                outputs.append(output)
                targets.append(target)
                errors.append(pjpe.mean(dim=1))
            else:
                break
            # UMAP
            if batch_idx < 10:
                zs.append(z)
                actions.append(batch['action'])
            else:
                break

    # # Poses Viz
    # outputs = torch.cat(outputs, 0)
    # targets = torch.cat(targets, 0)
    # errors = torch.cat(errors, 0)

    # plot_diffs(outputs, targets, errors, grid=4)

    # # UMAP
    # zs = torch.cat(zs, 0)
    # actions = torch.cat(actions, 0)
    # plot_umap(zs, actions)

    # mpjpe = torch.stack(pjpes, dim=0).mean(dim=0)
    mpjpe = torch.stack(pjpes, dim=0).sum(dim=0)/n_samples
    avg_mpjpe = torch.mean(mpjpe).item()

    config.logger.log({"MPJPE_AVG": avg_mpjpe})

    print(f'{vae_type} - * Mean MPJPE * : {round(avg_mpjpe,4)} \n {mpjpe}')

    del pjpes, mpjpe, zs, actions, norm_stats
    norm_stats.close()
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


def plot_umap(zs, actions):
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("[INFO] UMAP reducing ", zs.shape)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(zs)
    print(embedding.shape)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[
                sns.color_palette("husl", 17)[x] for x in actions.tolist()])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of Z', fontsize=24)
    plt.show()
