from .base import BaseCallback

class EarlyStopping(BaseCallback):
    def on_epoch_start(self, **kwargs):
        print("Early")