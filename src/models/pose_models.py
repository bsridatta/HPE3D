import torch.nn as nn
import torch

import torchvision.models as models


class Encoder2D(nn.Module):

    def __init__(self, latent_dim, n_joints=17, activation=nn.ReLU):
        super(Encoder2D, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_joints = n_joints
        self.__build_model()

    def __build_model(self):
        self.dense_block = nn.Sequential(
            nn.Linear(2*self.n_joints, 512),
            self.activation(),
            nn.Linear(512, 512),
            self.activation(),
            nn.Linear(512, 512),
            self.activation(),
        )
        self.fc_mean = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

    def forward(self, x):
        x = x.view(-1, 2*self.n_joints)
        x = self.dense_block(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Encoder3D(nn.Module):

    def __init__(self, latent_dim, n_joints=17, activation=nn.ReLU):
        super(Encoder3D, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_joints = n_joints
        self.__build_model()

    def __build_model(self):
        self.dense_block = nn.Sequential(
            nn.Linear(3*self.n_joints, 512),
            self.activation(),
            nn.Linear(512, 512),
            self.activation(),
            nn.Linear(512, 512),
            self.activation(),
        )
        self.fc_mean = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

    def forward(self, x):
        x = x.view(-1, 3*self.n_joints)
        x = self.dense_block(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder3D(nn.Module):
    def __init__(self, latent_dim, n_joints=17, activation=nn.ReLU):
        super(Decoder3D, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_joints = n_joints
        self.__build_model()

    def __build_model(self):
        self.dense_block = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            self.activation(),
            nn.Linear(512, 512),
            self.activation(),
            nn.Linear(512, 512),
            self.activation(),
            nn.Linear(512, 3*self.n_joints)
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        x = self.dense_block(x)
        return x


def reparameterize(mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mean + eps*std


def test():
    latent_dim = 2

    inp = torch.randn(2, 2, 17)  # [b, 2, joints]
    out = torch.randn(2, 3, 17)  # [b, 3, joints]

    encoder_2d = Encoder2D(latent_dim)
    encoder_3d = Encoder3D(latent_dim)
    decoder_3d = Decoder3D(latent_dim)
    encoder_2d.eval()
    encoder_3d.eval()
    decoder_3d.eval()

    # 2D -> 3D
    mean, logvar = encoder_2d(inp)
    z = reparameterize(mean, logvar)
    pose3d = decoder_3d(z)
    pose3d = pose3d.view(-1, 3, 17)
    loss = nn.functional.l1_loss(pose3d, out)
    print("2D -> 3D", loss)

    # 3D -> 3D
    mean, logvar = encoder_3d(out)
    z = reparameterize(mean, logvar)
    pose3d = decoder_3d(z)
    pose3d = pose3d.view(-1, 3, 17)
    loss = nn.functional.l1_loss(pose3d, out)
    print("2D -> 3D", loss)


if __name__ == "__main__":
    test()
