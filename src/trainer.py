from callback import Callbacks, EarlyStopping, ModelCheckpoint

if __name__=="__main__":
    early = EarlyStopping()
    ckpt = ModelCheckpoint()
    callbacks = Callbacks([early, ckpt])
    callbacks.on_epoch_start()