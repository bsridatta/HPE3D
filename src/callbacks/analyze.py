import torch
import numpy as np
import tensorflow as tf
import tensorboard as tb

from src.callbacks.base import Callback
from torch.utils.tensorboard import SummaryWriter
from src.viz.mpl_plots import plot_2d
from src.processing import post_process

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class Analyze(Callback):
    def __init__(self, checkpoint, n_samples=100):
        self.checkpoint = checkpoint
        self.n_samples = n_samples

    def setup(self, config, train_loader, val_loader, **kwargs):
        # Load appropriate checkpoint
        if config.resume_run is not None:
            print(f"[INFO]: Analyze Callback. using {self.checkpoint}")
            config.resume_run = self.checkpoint
            config.logger.config.update(config, allow_val_change=True, include_keys='resume_run')

        print("[INFO]: Analyze Callback. Skip training")
        train_loader.dataset.dataset_len = 0

        # Slice the validation datset
        val_loader.dataset.dataset_len = self.n_samples
        val_loader.shuffle = False

        print("Updated Validation Samples -", len(val_loader.dataset))

    def on_validation_end(self, config, t_data, **kwargs):
        writer = SummaryWriter(log_dir=config.logger.run.dir)

        # trynig to reuse the method
        t_data['input'], _ = post_process(config, t_data['input'], t_data['input'])

        images = []
        for i in t_data['input']:

            image_ = plot_2d(i.cpu().numpy(), mode='image', color='b')
            images.append(image_)

        images = torch.cat(images, 0)
        writer.add_embedding(t_data['z'], metadata=t_data['z_attr'], label_img=images)
        print(f"[INFO]: Analyze Callback. Saving results @ {config.logger.run.dir}")
        exit()
