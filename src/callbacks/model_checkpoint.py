from src.callbacks.base import Callback
import torch
import os


class ModelCheckpoint(Callback):
    def __init__(self):
        self.val_loss_min = float("inf")

    def setup(self, config, models, optimizers, variant, **kwargs):
        # Save model code to wandb
        config.logger.save(
            f"{os.path.dirname(os.path.abspath(__file__))}/models/pose*")

        # Resume training
        if config.resume_run != "None":
            # Models
            for model in list(models.values()):
                state = torch.load(
                    f'{config.save_dir}/{config.resume_run}_{model.name}.pt', map_location=config.device)
                print(
                    f'[INFO] Loaded Checkpoint {config.resume_run}: {model.name} @ epoch {state["epoch"]}')
                model.load_state_dict(state['model_state_dict'])

            # Optimizers
            for n_pair in range(len(variant)):
                optimizer_state_dic = torch.load(
                    f'{config.save_dir}/{config.resume_run}_optimizer_{n_pair}.pt', map_location=config.device)
                optimizers[n_pair].load_state_dict(optimizer_state_dic)

    def on_epoch_end(self, config, val_loss, model, optimizers, epoch, n_pair, **kwargs):
        if config.mpjpe < config.mpjpe_min and config.device.type != 'cpu':
            print(
                f"[INFO]: MPJPE decreased from {config.mpjpe_min} -> {config.mpjpe}")
            config.mpjpe_min = config.mpjpe  

            # just update val_loss for record
            if val_loss < self.val_loss_min and config.device.type != 'cpu':
                self.val_loss_min = val_loss

            # Models
            for model_ in model:
                try:
                    state_dict = model_.module.state_dict()
                except AttributeError:
                    state_dict = model_.state_dict()

                state = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'model_state_dict': state_dict,
                }

                torch.save(
                    state, f'{config.save_dir}/{config.logger.run.name}_{model_.name}.pt')
                config.logger.save(
                    f'{config.save_dir}/{config.logger.run.name}_{model_.name}.pt')
                print(
                    f'[INFO] Saved pt: {config.save_dir}/{config.logger.run.name}_{model_.name}.pt')

                del state

            # Optimizer
            torch.save(
                optimizers[n_pair].state_dict(),
                f'{config.save_dir}/{config.logger.run.name}_optimizer_{n_pair}.pt')
            config.logger.save(
                f'{config.save_dir}/{config.logger.run.name}_optimizer_{n_pair}.pt')
            print(
                f'[INFO] Saved pt: {config.save_dir}/{config.logger.run.name}_optimizer_{n_pair}.pt')
            
            # mpjpe_min corresponds to this model hence reproducible
            config.logger.config.update({"mpjpe_min": config.mpjpe_min}, allow_val_change=True)
