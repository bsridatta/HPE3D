from .base import BaseCallback

class ModelCheckpoint(BaseCallback):
    
    def on_train_start(self, **kwargs):
        # Resume training
        if config.resume_pt:
            logging.info(f'Loading {config.resume_pt}')
            state = torch.load(f'{config.save_dir}/{config.resume_pt}')
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
   
    def on_epoch_start(self, **kwargs):
        print("checkpoint")



        # if val_loss < val_loss_min:
        #     val_loss_min = val_loss
        #     try:
        #         # saving when model is in DP mode
        #         state_dict = model.module.state_dict()
        #     except AttributeError:
        #         # save model when not in DP or DDP
        #         state_dict = model.state_dict()

        #     state = {
        #         'epoch': epoch,
        #         'val_loss': val_loss,
        #         'model_state_dict': state_dict,
        #         'optimizer_state_dict': optimizer.state_dict()
        #     }
        #     torch.save(state, f'{config.save_dir}/{config.exp_name}.pt')
        #     logging.info(f'Saved pt: {config.save_dir}/{config.exp_name}.pt')

