import torch.nn as nn
import torch

import torchvision.models as models
'''
Reference (RGB Decoder) - https://github.com/spurra/vae-hands-3d
'''


class RGB_encoder(nn.Module):
    def __init__(self, latent_dim, pretrained=True, train_last_block=False):
        super(RGB_encoder, self).__init__()

        self.latent_dim = latent_dim
        self.pretrained = pretrained
        self.set_requires_grad = False if self.pretrained else True

        self.train_last_block = train_last_block
        # build model
        self.__build_model()

    def __build_model(self):
        backbone = models.resnet18(pretrained=self.pretrained)
        in_features = backbone.fc.in_features
        backbone = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*backbone)
        # Freeze all layers
        for name, param in self.backbone.named_parameters():
            param.requires_grad = self.set_requires_grad
            # To train more than just FCs
            if self.train_last_block and '7' in name:
                param.requires_grad = True  # the last conv block is 7.0 and 7.1

        # additional FC layers that output means and vars(log variance) of the distributions
        self.fc_mean = nn.Linear(in_features, self.latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class RGB_decoder(nn.Module):

    def __init__(self, latent_dim, activation=nn.ReLU):
        super(RGB_decoder, self).__init__()
        # latent space dimension (z)
        self.latent_dim = latent_dim
        # hidden/linear layer between latent space and deconv block
        self.hidden_dim = 256*8*8
        self.activation = activation

        self.__build_model()

    def __build_model(self):

        self.fc_block = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            self.activation()
        )
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            self.activation(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            self.activation(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            self.activation(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            self.activation(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1)
        )

    def forward(self, x):
        x = self.fc_block(x)
        x = x.view(-1, 256, 8, 8)
        x = self.conv_block(x)
        return x


def reparameterize(mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mean + eps*std


def test():
    latent_dim = 2

    inp = torch.randn(2, 3, 256, 256)  # [b, 3, w, h]
    image_dim = inp.shape[1:]

    rgb_encoder = RGB_encoder(latent_dim)
    rgb_decoder = RGB_decoder(latent_dim)
    rgb_encoder.eval()
    rgb_decoder.eval()

    mean, logvar = rgb_encoder(inp.float())
    z = reparameterize(mean, logvar)
    recon = rgb_decoder(z)

    assert(recon.shape[1:] == image_dim)
    print(nn.functional.l1_loss(recon, inp))


if __name__ == "__main__":
    test()
