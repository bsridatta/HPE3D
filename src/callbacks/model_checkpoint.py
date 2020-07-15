from src.callbacks.base import Callback
import torch

class ModelCheckpoint(Callback):
    
    # def on_train_start(self, **kwargs):
    #     # Resume training
    
    def on_epoch_end(self, config, val_loss, mpjpe, model, optimizer, epoch,**kwargs):
        if val_loss < config.val_loss_min:
            config.val_loss_min = val_loss
            
            config.mpjpe_at_min_val = mpjpe

            for model_ in model:
                try:
                    state_dict = model_.module.state_dict()
                except AttributeError:
                    state_dict = model_.state_dict()

                state = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                }
                # TODO save optimizer state seperately
                torch.save(
                    state, f'{config.save_dir}/{config.logger.run.name}_{model_.name}.pt')
                config.logger.save(
                    f'{config.save_dir}/{config.logger.run.name}_{model_.name}.pt')
                print(
                    f'[INFO] Saved pt: {config.save_dir}/{config.logger.run.name}_{model_.name}.pt')

                del state


