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
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def setup(self, config, train_loader, val_loader, **kwargs):
        # Load appropriate checkpoint
        print(f"[INFO]: Analyze Callback. using {self.checkpoint}")

        print("[INFO]: Analyze Callback. Skip training")
        train_loader.dataset.dataset_len = 0

        # Slice the validation datset
        val_loader.dataset.dataset_len = self.n_samples
        val_loader.shuffle = False
        print("Updated Validation Samples -", len(val_loader.dataset))
