from processing import scale_3d
from typing import Type
import torch
import torch.nn as nn


class LBAD(nn.Module):
    def __init__(
        self,
        neurons: int,
        activation: Type[torch.nn.Module],
        drop_out_p: float,
        use_bn: bool,
    ):
        super(LBAD, self).__init__()
        self.name = "LBAD"
        self.activ = activation()
        self.w1 = nn.Linear(neurons, neurons, bias=False)
        self.dropout = nn.Dropout(p=drop_out_p)
        self.bn1 = nn.BatchNorm1d(neurons)
        self.use_bn = use_bn
        
    def forward(self, x):
        x = self.w1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activ(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        BasicBlock,
        neurons: int,
        activation: Type[torch.nn.Module],
        drop_out_p: float,
        use_bn: bool = True,
    ):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.name = "ResBlock"
        # feature
        self.BB_1 = BasicBlock(neurons, activation, drop_out_p, use_bn)
        self.BB_2 = BasicBlock(neurons, activation, drop_out_p, use_bn)
        self.BB_3 = BasicBlock(neurons, activation, drop_out_p, use_bn)
        self.BB_4 = BasicBlock(neurons, activation, drop_out_p, use_bn)

    def forward(self, x):
        residual = x
        x = self.BB_1(x)
        x = self.BB_2(x) + residual

        residual = x
        x = self.BB_3(x)
        x = self.BB_4(x) + residual
        return x


class Encoder2D(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_joints: int,
        activation: Type[torch.nn.Module],
        neurons: int,
        drop_out_p: float,
    ):
        super(Encoder2D, self).__init__()
        self.name = "Encoder2D"
        self.n_joints = n_joints

        self.enc_inp_block = nn.Sequential(
            nn.Linear(2 * n_joints, neurons),
            nn.BatchNorm1d(neurons),
            activation(),
            nn.Dropout(p=drop_out_p),
        )

        self.features = ResBlock(LBAD, neurons, activation, drop_out_p)
        self.fc_mean = nn.Linear(neurons, latent_dim)
        self.fc_logvar = nn.Linear(neurons, latent_dim)
        # self.enc_out_block = nn.Sequential(
        #     nn.BatchNorm1d(self.latent_dim),
        #     self.activation(),
        # )

    def forward(self, x):
        x = x.view(-1, 2 * self.n_joints)
        x = self.enc_inp_block(x)
        x = self.features(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        # mean = self.enc_out_block(mean)
        # logvar = self.enc_out_block(logvar)
        return mean, logvar


class Decoder3D(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_joints: int,
        activation: Type[torch.nn.Module],
        neurons: int,
        drop_out_p: float,
    ):
        super(Decoder3D, self).__init__()
        self.name = "Decoder3D"
        self.latent_dim = latent_dim

        self.dec_inp_block = nn.Sequential(
            nn.Linear(self.latent_dim, neurons),
            nn.BatchNorm1d(neurons),
            activation(),
            nn.Dropout(p=drop_out_p),
        )
        self.features = ResBlock(LBAD, neurons, activation, drop_out_p)
        self.dec_out_block = nn.Sequential(
            nn.Linear(neurons, 3 * n_joints),
            nn.Tanh()
            # Shouldnt use BN for Generator output
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        x = self.dec_inp_block(x)
        x = self.features(x)
        x = self.dec_out_block(x)
        x = scale_3d(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        n_joints: int = 15,
        activation: Type[torch.nn.Module] = nn.LeakyReLU,
        neurons: int = 1024,
        drop_out_p: float = 0.5,
    ):
        super(Discriminator, self).__init__()
        self.name = "Discriminator"
        self.n_joints = n_joints

        self.inp_block = nn.Sequential(
            nn.Linear(2 * self.n_joints, neurons),
            activation(),
            # Shouldnt use BN for Critic input
        )
        self.features = ResBlock(LBAD, neurons, activation, drop_out_p, use_bn=False)
        self.out_block = nn.Sequential(nn.Linear(neurons, 1), nn.Sigmoid())

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        x = x.view(-1, 2 * self.n_joints)
        x = self.inp_block(x)
        x = self.features(x)
        x = self.out_block(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_joints: int = 15,
        activation: Type[torch.nn.Module] = nn.Mish,
        neurons: int = 1024,
        drop_out_p: float = 0.2,
    ):
        super(Generator, self).__init__()
        self.n_joints = n_joints
        self.encoder = Encoder2D(latent_dim, n_joints, activation, neurons, drop_out_p)
        self.decoder = Decoder3D(latent_dim, n_joints, activation, neurons, drop_out_p)

    @staticmethod
    def reparameterize(mean, logvar, is_eval=False):
        if is_eval:  # TODO double check this
            return mean

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, max=30)  # to prevent exp(logvar) -> inf
        z = self.reparameterize(mean, logvar, is_eval=not self.encoder.training)
        recon = self.decoder(z)
        return recon.view(-1, self.n_joints, 3), mean, logvar
