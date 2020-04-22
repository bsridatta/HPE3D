from callbacks import CallbackList, EarlyStopping, ModelCheckpoint

if __name__=="__main__":
    early = EarlyStopping()
    ckpt = ModelCheckpoint()
    callbacks = CallbackList([early, ckpt])
    callbacks.on_epoch_start()