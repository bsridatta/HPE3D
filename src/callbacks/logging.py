from src.callbacks.base import Callback
import torch
from src.models import PJPE


class Logging(Callback):
    """Logging and printing metrics"""

    def setup(self, config, models, **kwargs):
        print(
            f'[INFO]: Start training procedure using device: {config.device}')
            
        for model in models.values():
            config.logger.watch(model, log='all')

    def on_train_batch_end(self, config, vae_type, epoch, batch_idx, batch, dataloader, output, **kwargs):
        # wandb
        config.logger.log({
            f"{vae_type}": {
                "train": {
                    "kld_loss": output['log']['kld_loss'],
                    "recon_loss": output['log']['recon_loss'],
                    "total_train": output['loss']
                }
            }
        })

        # print to console
        batch_len = len(batch['pose2d'])
        dataset_len = len(dataloader.dataset)
        n_batches = len(dataloader)
        print('\n {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tReCon: {:.4f}\tKLD: {:4f}'.format(
            vae_type, epoch, batch_idx * batch_len,
            dataset_len, 100. *
            batch_idx / n_batches,
            output['loss'], output['log']['recon_loss'], output['log']['kld_loss']), end='')

        if config.self_supervised:
            print('\tCritic: {:.4f}'.format(output['log']['critic_loss']), end='')
            
            config.logger.log({
                f"{vae_type}": {
                    "train": {
                        "critic_loss": output['log']['critic_loss']
                    }
                }
            })


    def on_validation_end(self, config, vae_type, epoch, critic_loss, avg_loss, recon_loss, kld_loss, val_loader, mpjpe, pjpe, **kwargs):
        # average epochs output
        avg_output = {}
        avg_output['loss'] = avg_loss
        avg_output['log'] = {}
        avg_output['log']['recon_loss'] = recon_loss/len(val_loader)
        avg_output['log']['kld_loss'] = kld_loss/len(val_loader)

        # log to wandb
        config.logger.log({
            f"{vae_type}": {
                "val": {
                    "kld_loss": avg_output['log']['kld_loss'],
                    "recon_loss": avg_output['log']['recon_loss'],
                    "total_val": avg_output['loss']
                }
            }
        })

        # print to console
        print(f"\n{vae_type} Validation:",
              f"\t\tLoss: {round(avg_loss,4)}",
              f"\tReCon: {round(avg_output['log']['recon_loss'], 4)}",
              f"\tKLD: {round(avg_output['log']['kld_loss'], 4)}", end='')

        if config.self_supervised:
            avg_output['log']['critic_loss'] = critic_loss/len(val_loader)
            print(f"\tCritic: {round(avg_output['log']['critic_loss'], 4)}", end='')
    
            config.logger.log({
                f"{vae_type}": {
                    "val": {
                        "critic_loss": avg_output['log']['critic_loss']
                    }
                }
            })
        
        # print and log MPJPE
        print(f'\n{vae_type} - * MPJPE * : {round(mpjpe,4)} \n {pjpe}')
        config.logger.log({f'{vae_type}_mpjpe': mpjpe})
        config.mpjpe = mpjpe

        if mpjpe < config.mpjpe_min:
            config.mpjpe_min = mpjpe

        # For Images
        # TODO can have this in eval instead and skip logging val
        # if output['epoch'] % 2 == 0 and "rgb" in vae_type.split('_')[-1]:
        #     # TODO change code to wandb
        #     config.logger.log(
        #         f"Images/{output['epoch']}", output['recon'][0])
