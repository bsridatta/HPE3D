from typing import Type
import pytorch_lightning as pl
import torch.nn.functional as F
from models import Discriminator, Generator
import torch
from processing import translate_and_project, random_rotate, scale_3d


class VAEGAN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lr_g: float,
        lr_d: float,
        is_ss: bool = True,
        project_dist: float = 10,
    ) -> None:
        super().__init__()
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.is_ss = is_ss
        self.project_dist = project_dist

        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        self.real = 1
        self.fake = 0

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        """
        # TODO add miss joint augmentation, noise for disc. training
        """
        if not self.is_ss:
            return self.training_step_supervised(batch)

        inp, target = batch["pose2d"], batch["pose2d"]
        recon_3d, mean, logvar = self.generator(inp)
        recon_3d = scale_3d(recon_3d)
        recon_2d = translate_and_project(recon_3d, self.project_dist)  # for recon loss
        novel_2d = translate_and_project(random_rotate(recon_3d), self.project_dist)  # to train G

        opt_g, opt_d = self.optimizers()
        labels = torch.full((len(inp), 1), self.real).to(inp.device).type_as(inp)

        """Train D"""
        opt_d.zero_grad(set_to_none=True)
        # with real
        out = self.discriminator(inp)
        loss_d_real = self.adversarial_loss(out, labels.fill_(self.real))
        self.manual_backward(loss_d_real)
        D_x = out.mean().item()
        # with fake
        out = self.discriminator(novel_2d.detach()) 
        loss_d_fake = self.adversarial_loss(out, labels.fill_(self.fake))
        self.manual_backward(loss_d_fake)
        D_G_z1 = out.mean().item()
        loss_d = loss_d_real + loss_d_fake
        opt_d.step()

        """Train G"""
        opt_g.zero_grad(set_to_none=True)
        # with same fake/ novel_2d sample
        out = self.discriminator(novel_2d)
        loss_g = self.adversarial_loss(out, labels.fill_(self.real))
        self.manual_backward(loss_g)
        D_G_z2 = out.mean().item()
        opt_g.step()

    def training_step_supervised(self, batch):
        inp, target = batch["pose2d"], batch["pose3d"]
        recon_3d, mean, logvar = self.generator(inp)
        loss = self.recon_loss(recon_3d, target) + self.kld_loss(mean, logvar)
        return loss

    def adversarial_loss(self, y_hat, y, reduction: str = "mean"):
        return F.binary_cross_entropy(y_hat, y, reduction=reduction)

    def recon_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def kld_loss(self, mean, logvar):
        # mean as recon is mean or reduction is sum for both TODO i guess
        return torch.mean(
            -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0
        )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999)
        )
        opt_d = optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999)
        )  # using SGD worsens Dx

        return [opt_g, opt_d], []  # TODO scheduler
