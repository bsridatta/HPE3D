import torch.nn.functional as F
from collections import OrderedDict 
import torch

def training_epoch(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device) if key != 'image_id' else batch[key]
               
        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model)
        _log_training_metrics(args, output)
        loss = output['loss_val']
        loss.backward()
        optimizer.step()
        

        if batch_idx % 100 == 0:
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            state = {
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state, f'{args.save_dir}/_backup_{args.exp_name}.pt')
   

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch['image_id']), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))   
        
        del loss
