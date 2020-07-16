from src.callbacks.base import Callback
import torch


class MaxNorm(Callback):
    '''Max norm constraint. Regularization to prevent norm of weights going beyond 1
    '''
    def on_train_batch_end(self, models, max_val=1, eps=1e-8, **kwargs):
        """
        clip the norm of the weights to 1, as suggested in Martinez et. al
        Performed after update step

        Args:
            models (nn.Model): pytorch models, Encoder and Decoder
            max_val (int): max norm constraint value. Defaults to 1.
            eps (float): To avoid nan division by zero. Defaults to 1e-8.
        """
        for model in models:
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    norm = param.norm(2, dim=0, keepdim=True)
                    desired = torch.clamp(norm, 0, max_val)
                    param = param * (desired / (eps + norm))
            