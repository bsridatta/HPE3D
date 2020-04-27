import torch.nn.functional as F
from collections import OrderedDict
import torch
from models import reparameterize, KLD, MPJPE
import utils


def training_epoch(config, model, train_loader, optimizer, epoch):
    # model.train() inside training step
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).long()
        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model)
        _log_training_metrics(config, output)
        loss = output['loss_val']
        loss.backward()
        optimizer.step()
        print("[trainer] loss", loss.item())

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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                len(batch['pose2d']), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        del loss


def validation_epoch(config, model, val_loader):
    # model.eval() in validation step
    loss = 0
    acc = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(config.device).long()

            output = _validation_step(batch, batch_idx, model)
            _log_validation_metrics(config, output)
            loss += output['loss_val'].item()

    avg_loss = loss/len(val_loader)
    print(f'Val set: Average Loss: {avg_loss}')

    return avg_loss


def _training_step(batch, batch_idx, model):
    encoder = model[0]
    decoder = model[1]
    encoder.train()
    decoder.train()

    inp, target = utils.get_inp_target(encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    output = decoder(z)
    output = output.view(target.shape)

    recon_loss = MPJPE(output, target)
    kld_loss = KLD(mean, logvar)
    loss_val = kld_loss+recon_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss,
                   "total_loss": loss_val}

    return OrderedDict({'loss_val': loss_val, 'log': logger_logs})


def _validation_step(batch, batch_idx, model):
    encoder = model[0]
    decoder = model[1]
    encoder.eval()
    decoder.eval()

    inp, target = utils.get_inp_target(encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    output = decoder(z)
    output = output.view(target.shape)

    recon_loss = MPJPE(output, target)
    kld_loss = KLD(mean, logvar)
    loss_val = kld_loss+recon_loss

    logger_logs = {"kld_loss": kld_loss,
                   "recon_loss": recon_loss,
                   "total_loss": loss_val}

    return OrderedDict({'loss_val': loss_val, "log": logger_logs})


def _log_training_metrics(config, output):
    config.writer.add_scalars(f"Loss/Train_Loss", output['log'], 0)
    config.writer.add_scalar("Total/Train_Loss", output['loss_val'])


def _log_validation_metrics(config, output):
    config.writer.add_scalars(f"Loss/Val_Loss", output['log'], 0)
    config.writer.add_scalar("Total/Val_Loss", output["loss_val"])
