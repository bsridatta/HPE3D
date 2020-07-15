import unittest

import torch
import torchvision.models as models

from src.models import pose_models, rgb_models
from src.models.model_utils import KLD, PJPE, reparameterize


class ModelTestCase(unittest.TestCase):

    def test_pose_models(self):
        inp = torch.randn(4, 16, 2)  # [batch_size, joints, 2]
        target = torch.randn(4, 16, 3)  # [batch_size, joints, 3]

        latent_dim = 2

        encoder_2d = pose_models.Encoder2D(latent_dim)
        decoder_3d = pose_models.Decoder3D(latent_dim)

        encoder_2d.eval()
        decoder_3d.eval()

        with torch.no_grad():
            mean, logvar = encoder_2d(inp)
            z = reparameterize(mean, logvar)
            pose3d = decoder_3d(z)

            pose3d = pose3d.view(-1, 16, 3)
            recon_loss = PJPE(pose3d, target)
            kld_loss = KLD(mean, logvar, decoder_3d.__class__.__name__)
            print("2D -> 3D", recon_loss)

    def test_rgb_models(self):
        inp = torch.randn(2, 3, 256, 256)  # [b, 3, w, h]
        image_dim = inp.shape[1:]

        latent_dim = 2

        encoder_RGB = rgb_models.EncoderRGB(latent_dim)
        decoder_RGB = rgb_models.DecoderRGB(latent_dim)

        encoder_RGB.eval()
        decoder_RGB.eval()

        mean, logvar = encoder_RGB(inp.float())
        z = reparameterize(mean, logvar)
        recon = decoder_RGB(z)

        self.assertEqual(recon.shape[1:], image_dim)
        print(torch.nn.functional.l1_loss(recon, inp))


if __name__ == "__main__":
    unittest.main()
