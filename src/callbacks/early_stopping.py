from .base import Callback

class EarlyStopping(Callback):
    def on_epoch_start(self, **kwargs):
        print("Early")