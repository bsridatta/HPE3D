import os

import torch

from src.callbacks.base import Callback
from src.models import PJPE
from src.viz.mpl_plots import plot_all_proj, plot_3d
from src.viz.mayavi_plots import plot_3D_models


class Logging(Callback):
    """Logging and printing metrics"""

    def setup(self, config, models, **kwargs):
        print(
            f'[INFO]: Start training procedure using device: {config.device}')

        for model in models.values():
            config.logger.watch(model, log='all')

    def on_train_batch_end(self, config, vae_type, epoch, batch_idx, batch, dataloader, output, **kwargs):
        # print to console
        batch_len = len(batch['pose2d'])
        dataset_len = len(dataloader.dataset)
        n_batches = len(dataloader)
        print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tReCon: {:.4f}\tKLD: {:4f}'.format(
            vae_type, epoch, batch_idx * batch_len,
            dataset_len, 100. *
            batch_idx / n_batches,
            output['loss'], output['log']['recon_loss'], output['log']['kld_loss']), end='')

        # critic
        if config.self_supervised:
            print('\tCritic: {:.4f}\tCritic Real: {:.4f}\tCritic Fake: {:.4f}'.format(
                output['log']['critic_loss'], output['log']['critic_loss_real'],
                output['log']['critic_loss_fake']), end='')

            config.logger.log({
                f"{vae_type}": {
                    "train": {
                        "critic_loss": output['log']['critic_loss'],
                        "critic_loss_real": output['log']['critic_loss_real'],
                        "critic_loss_fake": output['log']['critic_loss_fake']
                    }
                }
            }, commit=True)

            # TODO change -1 to 0 to log images
            if (batch_idx/n_batches) % 0.1 == 0 and batch_len == config.batch_size:
                i = 0
                plot_all_proj(config, output["log"]["recon_2d"][i], output["log"]["novel_2d"][i], output["log"]["target_2d"][i],
                              output["log"]["recon_3d"][i], output["log"]["target_3d"][i])

        # other logs to wandb
        config.logger.log({
            f"{vae_type}": {
                "train": {
                    "kld_loss": output['log']['kld_loss'],
                    "recon_loss": output['log']['recon_loss'],
                    "total_train": output['loss']
                }
            }
        }, commit=True)

        print('')

    def on_validation_start(self):
        print("Start validation epoch")

    def on_validation_end(self, config, vae_type, epoch, loss_dic, val_loader, mpjpe, avg_pjpe, pjpe, t_data, **kwargs):
        # average epochs output
        avg_output = {}
        avg_output['log'] = {}

        avg_output['loss'] = loss_dic['loss']/len(val_loader)
        avg_output['log']['recon_loss'] = loss_dic['recon_loss']/len(val_loader)
        avg_output['log']['kld_loss'] = loss_dic['kld_loss']/len(val_loader)

        # print to console
        print(f"{vae_type} Validation:",
              f"\t\tLoss: {round(avg_output['loss'],4)}",
              f"\tReCon: {round(avg_output['log']['recon_loss'], 4)}",
              f"\tKLD: {round(avg_output['log']['kld_loss'], 4)}", end='')

        # critic
        if config.self_supervised:
            avg_output['log']['critic_loss'] = loss_dic['critic_loss']/len(val_loader)
            avg_output['log']['critic_loss_real'] = loss_dic['critic_loss_real']/len(val_loader)
            avg_output['log']['critic_loss_fake'] = loss_dic['critic_loss_fake']/len(val_loader)

            print('\tCritic: {:.4f}\tCritic Real: {:.4f}\tCritic Fake: {:.4f}'.format(
                avg_output['log']['critic_loss'], avg_output['log']['critic_loss_real'],
                avg_output['log']['critic_loss_fake'], end=''))

            # log to wandb
            config.logger.log({
                f"{vae_type}": {
                    "val": {
                        "critic_loss": avg_output['log']['critic_loss'],
                        "critic_loss_real": avg_output['log']['critic_loss_real'],
                        "critic_loss_fake": avg_output['log']['critic_loss_fake']
                    }
                }
            }, commit=True)

            # log intermediate visualizations
            n_samples = 3
            for i in range(n_samples):
                i+=round(len(t_data["recon_2d"])/4.2)
                plot_all_proj(config, t_data["recon_2d"][i], t_data["novel_2d"][i], t_data["target_2d"][i],
                              t_data["recon_3d"][i], t_data["target_3d"][i], name='val', title=pjpe[i].item())

        # log main metrics to wandb
        config.logger.log({
            f"{vae_type}": {
                "val": {
                    "kld_loss": avg_output['log']['kld_loss'],
                    "recon_loss": avg_output['log']['recon_loss'],
                    "total_val": avg_output['loss']
                }
            }
        }, commit=True)

        # print and log MPJPE
        print(f'{vae_type} - * MPJPE * : {round(mpjpe,4)} \n {avg_pjpe}')
        config.logger.log({f'{vae_type}_mpjpe': mpjpe})
        config.mpjpe=mpjpe

        if mpjpe < config.mpjpe_min:
            config.mpjpe_min=mpjpe

        # For Images
        # TODO can have this in eval instead and skip logging val
        # if output['epoch'] % 2 == 0 and "rgb" in vae_type.split('_')[-1]:
        #     # TODO change code to wandb
        #     config.logger.log(
        #         f"Images/{output['epoch']}", output['recon'][0])
