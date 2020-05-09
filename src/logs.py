import torch


def _log_training_metrics(config, output, vae_type):
    config.writer.add_scalars(f"Loss/{vae_type}/Train_Loss", output['log'], 0)
    config.writer.add_scalar(
        f"Total/{vae_type}/Train_Loss", output['loss_val'])


def _log_validation_metrics(config, output, vae_type):
    if output['epoch'] % 2 == 0 and "rgb" in vae_type.split('_')[-1]:
        config.writer.add_image(
            f"Images/{output['epoch']}", output['recon'][0])
    config.writer.add_scalars(f"Loss/{vae_type}/Val_Loss", output['log'], 0)
    config.writer.add_scalar(f"Total/{vae_type}/Val_Loss", output["loss_val"])


def save_model(config, model, optimizer, type):
    try:
        # model when in DP mode
        state_dict = model.module.state_dict()
    except AttributeError:
        # model when not in DP or DDP mode
        state_dict = model.state_dict()

    state = {
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, f'{config.save_dir}/_backup_{config.exp_name}.pt')
