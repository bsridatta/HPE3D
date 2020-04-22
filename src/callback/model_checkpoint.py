from .base import BaseCallback

class ModelCheckpoint(BaseCallback):
    def on_epoch_start(self, **kwargs):
        print("checkpoint")

