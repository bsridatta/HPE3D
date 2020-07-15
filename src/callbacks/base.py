'''
Callback inspritations from PyTorch Lightning - https://github.com/PyTorchLightning/PyTorch-Lightning
and https://github.com/devforfu/pytorch_playground/blob/master/loop.ipynb
'''

import abc


class Callback(abc.ABC):
    def on_epoch_start(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_batch_start(self, **kwargs): pass
    def on_batch_end(self, **kwargs): pass
    def on_train_start(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass
    def on_validation_start(self, **kwargs): pass
    def on_validation_end(self, **kwargs): pass
    def on_test_start(self, **kwargs): pass
    def on_test_end(self, **kwargs): pass


class CallbackList(Callback):

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_start(self, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_start(**kwargs)

    def on_epoch_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)

    def on_train_start(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_start(**kwargs)

    def on_train_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_batch_start(self, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_start(**kwargs)

    def on_batch_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(**kwargs)

    def on_validation_start(self, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_start(**kwargs)

    def on_validation_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_end(**kwargs)

    def on_test_start(self, **kwargs):
        for callback in self.callbacks:
            callback.on_test_start(**kwargs)

    def on_test_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_test_end(**kwargs)
