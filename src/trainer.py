import torch.nn.functional as F
from collections import OrderedDict
import torch
from models import reparameterize, KLD, MPJPE
import utils


def training_epoch(config, model, train_loader, optimizer, epoch):

    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(config.device).long()
        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model)
        # _log_training_metrics(config, output)
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
        #     torch.save(state, f'{args.save_dir}/_backup_{args.exp_name}.pt')

        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(batch['image_id']), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

        del loss


def validation_epoch(args, model, val_loader):
    model.eval()
    loss = 0
    acc = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(
                    args.device) if key != 'image_id' else batch[key]

            output = _validation_step(batch, batch_idx, model)
            _log_validation_metrics(args, output)
            loss += output['val_loss'].item()
            acc += output['val_acc']

    avg_loss = loss/len(val_loader)
    avg_acc = acc/len(val_loader)
    print(f'Val set: Average Loss: {avg_loss}, Accuracy: {avg_acc}')

    return avg_loss


def _training_step(batch, batch_idx, model):
    encoder = model[0]
    decoder = model[1]
    inp, target = utils.get_inp_target(encoder, decoder, batch)
    mean, logvar = encoder(inp)
    z = reparameterize(mean, logvar)
    output = decoder(z)
    output = output.view(target.shape)

    recon_loss = MPJPE(output, target)
    
    kld_loss = KLD(mean, logvar)
    loss_val = kld_loss+recon_loss
    logger_logs = []
    return OrderedDict({'loss_val': loss_val, 'log': logger_logs})


def _validation_step(batch, batch_idx, model):
    grapheme, vowel, consonant = model(batch['image'])

    loss_grapheme = F.cross_entropy(grapheme, batch['grapheme_root'].long())
    loss_vowel = F.cross_entropy(vowel, batch['vowel_diacritic'].long())
    loss_consonant = F.cross_entropy(
        consonant, batch['consonant_diacritic'].long())
    loss_val = loss_grapheme + loss_vowel + loss_consonant

    acc_grapheme = torch.sum(grapheme.argmax(
        dim=1) == batch["grapheme_root"]).item() / (len(grapheme) * 1.0)
    acc_vowel = torch.sum(vowel.argmax(dim=1) ==
                          batch["vowel_diacritic"]).item() / (len(vowel) * 1.0)
    acc_consonant = torch.sum(consonant.argmax(
        dim=1) == batch["consonant_diacritic"]).item() / (len(consonant) * 1.0)
    val_acc = acc_grapheme + acc_vowel + acc_consonant
    val_acc = val_acc/3
    logger_logs = {"VLoss_G": loss_grapheme,
                   "VLoss_V": loss_vowel,
                   "VLoss_C": loss_consonant,
                   "VAcc_G": acc_grapheme,
                   "VAcc_V": acc_vowel,
                   "VAcc_C": acc_consonant,
                   "VAcc": val_acc,
                   "VLoss": loss_val
                   }

    return OrderedDict({'val_loss': loss_val, 'val_acc': val_acc, 'log': logger_logs})


def _log_training_metrics(args, output):
    logs = output['log']
    split_loss = dict(filter(lambda item: "TLoss_" in item[0], logs.items()))
    args.writer.add_scalars(f"Loss/Train_Loss", split_loss, 0)

    args.writer.add_scalar("Total/Train_Loss", logs["TLoss"])


def _log_validation_metrics(args, output):
    logs = output['log']
    loss = dict(filter(lambda item: "VLoss_" in item[0], logs.items()))
    args.writer.add_scalars(f"Loss/Val_Loss", loss)
    args.writer.add_scalar("Total/Val_Loss", logs["VLoss"])

    acc = dict(filter(lambda item: "VAcc_" in item[0], logs.items()))
    args.writer.add_scalars(f"Acc/Val_Acc", acc)
    args.writer.add_scalar("Total/Val_Acc", logs["VAcc"])
