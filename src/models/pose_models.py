import torch.nn as nn
import torch

import torchvision.models as models
from data import plot_h36

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


def MPJPE(pred, target):
    # per sample in batch
    # mean(root(x2+y2+z2)) mean(PJPE)
    mpjpe = torch.mean(torch.sqrt(torch.sum((pred-target).pow(2), dim=2)), dim=1)
    # reduction = mean of mpjpe across batch
    return torch.mean(mpjpe)


def KLD(mean, logvar):
    loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return loss


def test(inp, target):
    latent_dim = 2

    # inp = torch.randn(2, 17, 2)  # [b, joints. 2]
    # target = torch.randn(2, 17, 3)  # [b, joints, 3]

    encoder_2d = Encoder2D(latent_dim)
    encoder_3d = Encoder3D(latent_dim)
    decoder_3d = Decoder3D(latent_dim)
    encoder_2d.eval()
    encoder_3d.eval()
    decoder_3d.eval()

    with torch.no_grad():

        # 2D -> 3D
        mean, logvar = encoder_2d(inp)
        z = reparameterize(mean, logvar)
        pose3d = decoder_3d(z)
        pose3d = pose3d.view(-1, 17, 3)
        recon_loss = MPJPE(pose3d, target)
        # loss = nn.functional.l1_loss(pose3d, target)
        kld_loss = KLD(mean, logvar)
        print("2D -> 3D", recon_loss)
        plot_h36(pose3d)
        exit()
        # 3D -> 3D
        mean, logvar = encoder_3d(target)
        z = reparameterize(mean, logvar)
        pose3d = decoder_3d(z)
        pose3d = pose3d.view(-1, 17, 3)
        loss = nn.functional.l1_loss(pose3d, target)
        print("3D -> 3D", loss)


if __name__ == "__main__":

    pose2d = [[473.68356, 444.9424],
              [500.9961, 448.02988],
              [479.83926, 530.78564],
              [506.21838, 622.56885],
              [445.9001, 441.81586],
              [456.18906, 537.1581],
              [467.30923, 633.76935],
              [488.18674, 397.43405],
              [481.02847, 340.39694],
              [478.51755, 318.808],
              [485.76895, 297.57162],
              [454.01608, 359.75955],
              [430.05878, 415.7349],
              [412.99722, 452.88666],
              [508.13437, 356.49152],
              [520.3154, 413.31827],
              [515.4715, 456.42984]]

    pose3d = [[-176.73077, -321.0486, 5203.882],
              [-52.96191, -309.7045, 5251.083],
              [-155.64156,   73.071754, 5448.807],
              [-29.831573,  506.78445, 5400.138],
              [-300.49985, -332.39276, 5156.681],
              [-258.24048,   99.60905, 5244.6816],
              [-209.48436,  548.8338, 5290.7637],
              [-109.15762, -529.72815, 5123.8906],
              [-140.19118, -780.1214, 5074.605],
              [-153.18188, -886.97614, 5130.1655],
              [-118.93483, -970.2283, 5058.599],
              [-259.08997, -690.1336, 5050.5923],
              [-370.6709, -448.5993, 5134.1772],
              [-462.28662, -290.82947, 5307.6274],
              [-19.760342, -716.9181, 5140.2725],
              [35.79161, -470.14493, 5257.7383],
              [13.892465, -279.85294, 5421.0684]]

    test(torch.FloatTensor([pose2d]), torch.FloatTensor([pose3d]))
