from utils import auto_init_args
from typing import Tuple, Type
import pytorch_lightning as pl
import torch.nn.functional as F
from models import Discriminator, Generator
import torch
from processing import translate_and_project, random_rotate, scale_3d


class VAEGAN(pl.LightningModule):
    def __init__(
        self,
        opt,
        **kwargs,
    ) -> None:
        super().__init__()
        auto_init_args(self)
        self.opt = opt
        self.generator = Generator(opt.latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        self.project_dist = 10

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        """TODO add miss joint augmentation, noise for disc. training"""
        if not self.opt.is_ss:
            return self.supervised_step(batch)

        inp, target = batch["pose2d"], batch["pose2d"]
        recon_3d, mean, logvar = self.generator(inp)
        recon_2d = translate_and_project(recon_3d, self.project_dist)
        loss_recon = self.recon_loss(recon_2d, target)
        loss_kld = self.kld_loss(mean, logvar)

        novel_2d = translate_and_project(
            random_rotate(recon_3d), self.project_dist
        )  # fakes to train G and D

        opt_g, opt_d = self.optimizers()
        reals, fakes = self.get_label(inp, )
        """Train D"""
        opt_d.zero_grad(set_to_none=True)
        # with real
        out = self.discriminator(inp)
        loss_d_real = self.adversarial_loss(out, reals)
        self.manual_backward(loss_d_real)
        D_x = out.mean().item()
        # with fake
        out = self.discriminator(novel_2d.detach())
        loss_d_fake = self.adversarial_loss(out, fakes)
        self.manual_backward(loss_d_fake)
        D_G_z1 = out.mean().item()
        loss_d = loss_d_real + loss_d_fake
        opt_d.step()

        """Train G"""
        opt_g.zero_grad(set_to_none=True)
        # with same fake/ novel_2d sample
        out = self.discriminator(novel_2d)
        loss_g = self.adversarial_loss(out, reals)  # includes Enc.
        loss_vae = loss_g + loss_recon + loss_kld  # G -> realistic + proj recon acc.
        self.manual_backward(loss_vae)
        D_G_z2 = out.mean().item()
        opt_g.step()

    def supervised_step(self, batch):
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
            self.generator.parameters(), lr=self.opt.lr_g, betas=(0.5, 0.999)
        )
        opt_d = optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.opt.lr_d, betas=(0.5, 0.999)
        )  # using SGD worsens Dx

        return [opt_g, opt_d], []  # TODO scheduler

    def get_label(self, inp: torch.Tensor, smooth: bool=True):
        """label smoothing for real labels 
           TODO flip labels - confuse D.
        """
        real = 1
        fake = 0
        reals = torch.full((len(inp), 1), real).to(inp.device).type_as(inp)
        fakes = reals.fill_(fake)
        if smooth:
            noise = (torch.rand_like(reals)*(0.7-1.2)) + 1.2
            reals = reals * noise.to(reals.device).type_as(reals)
        return reals, fakes