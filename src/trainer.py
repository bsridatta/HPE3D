import torch.nn.functional as F
from collections import OrderedDict
import torch
from models import reparameterize, KLD, MPJPE
import utils


def training_epoch(config, model, train_loader, optimizer, epoch, vae_type):
    # model.train() inside training step
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).long()
        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model)
        _log_training_metrics(config, output, vae_type)
        loss = output['loss_val']
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
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                vae_type, epoch, batch_idx *
                len(batch['pose2d']), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        del loss


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
    print(f'{vae_type} - Val set: Average Loss: {avg_loss}')

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
    loss_val = kld_loss+recon_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss_val': loss_val, 'log': logger_logs})


def _validation_step(batch, batch_idx, model, epoch):
    encoder = model[0]
    decoder = model[1]
    encoder.eval()
    decoder.eval()

    inp, target, criterion = utils.get_inp_target_criterion(encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    output = decoder(z)
    output = output.view(target.shape)
    recon_loss = criterion(output, target)
    kld_loss = KLD(mean, logvar)
    loss_val = kld_loss+recon_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss}

    return OrderedDict({'loss_val': loss_val, "log": logger_logs,  "recon": output,
                        "epoch": epoch})


def _log_training_metrics(config, output, vae_type):
    config.writer.add_scalars(f"Loss/{vae_type}/Train_Loss", output['log'], 0)
    config.writer.add_scalar(
        f"Total/{vae_type}/Train_Loss", output['loss_val'])


def _log_validation_metrics(config, output, vae_type):
    if output['epoch'] % 2 == 0 and "rgb" in vae_type.split('_')[-1]:
        config.writer.add_image(f"Images/{output['epoch']}", output['recon'][0])
    config.writer.add_scalars(f"Loss/{vae_type}/Val_Loss", output['log'], 0)
    config.writer.add_scalar(f"Total/{vae_type}/Val_Loss", output["loss_val"])
