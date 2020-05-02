import logging
import os
import sys
from argparse import ArgumentParser

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from pose_models import reparameterize, Encoder2D, Encoder3D, Decoder3D
from rgb_models import EncoderRGB, DecoderRGB


def do_forward(encoder, decoder, inp, target):
    model_params = list(encoder.parameters())+list(decoder.parameters())
    optimizer = torch.optim.Adam(model_params, lr=0.0001)

    p = []
    for x in range(2):
        optimizer.zero_grad()
        mean, logvar = encoder(inp)
        z = reparameterize(mean, logvar)
        pose3d = decoder(z)
        pose3d = pose3d.view(-1, 3, 17)
        loss = nn.functional.l1_loss(target, pose3d)
        print("2D -> 3D", loss)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    latent_dim = 2

    inp = torch.randn(2, 3, 256, 256)  # [b, 3, w, h]

    inp = torch.randn(2, 2, 17)  # [b, 2, joints]
    out = torch.randn(2, 3, 17)  # [b, 3, joints]

    # Models
    encoder_RGB = EncoderRGB(latent_dim)
    encoder2D = Encoder2D(latent_dim)
    decoder_RGB = DecoderRGB(latent_dim)
    decoder3D = Decoder3D(latent_dim)
    encoder_RGB.train()
    encoder2D.train()
    decoder_RGB.train()
    decoder3D.train()

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 2D -> 3D
    do_forward(encoder2D, decoder3D, inp, out)
    exit()
    # 3D -> 3D
    mean, logvar = encoder_3d(out)
    z = reparameterize(mean, logvar)
    pose3d = decoder_3d(z)
    pose3d = pose3d.view(-1, 3, 17)
    loss = nn.functional.l1_loss(pose3d, out)
