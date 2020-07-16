from src.callbacks.base import Callback
import torch

class ModelCheckpoint(Callback):
    
    def setup(self, config, models, optimizers, **kwargs):
        # Resume training   
        if config.resume_run not in "None":
            for vae in range(len(models)):
                for model_ in models[vae]:
                    state = torch.load(
                        f'{config.save_dir}/{config.resume_run}_{model_.name}.pt', map_location=config.device)
                    print(
                        f'[INFO] Loaded Checkpoint {config.resume_run}: {model_.name} @ epoch {state["epoch"]}')
                    model_.load_state_dict(state['model_state_dict'])
                    optimizers[vae].load_state_dict(state['optimizer_state_dict'])
                    # TODO load optimizer state seperately w.r.t variant
    
    def on_epoch_end(self, config, val_loss, model, optimizer, epoch,**kwargs):
        # Save if doing some real training
        if val_loss < config.val_loss_min and config.device!='cpu':
            config.val_loss_min = val_loss

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


