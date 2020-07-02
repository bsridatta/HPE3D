import sys

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder2D(nn.Module):

    def __init__(self, latent_dim, n_joints=16, activation=nn.ReLU):
        super(Encoder2D, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_joints = n_joints
        self.neurons = 512
        self.name = "Encoder2D"
        self.drop_out_p = 0

        self.__build_model()

    def __build_model(self):

        self.enc_inp_block = nn.Sequential(
            nn.Linear(2*self.n_joints, self.neurons),  # expand features
            nn.BatchNorm1d(self.neurons),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )

        self.fc_mean = nn.Linear(self.neurons, self.latent_dim)
        self.fc_logvar = nn.Linear(self.neurons, self.latent_dim)

        self.enc_out_block = nn.Sequential(
            nn.BatchNorm1d(self.latent_dim),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )

        self.LBAD_block = nn.Sequential(
            nn.Linear(self.neurons, self.neurons),
            nn.BatchNorm1d(self.neurons),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )

        self.LA_block = nn.Sequential(
            nn.Linear(self.neurons, self.neurons),
            self.activation()
        )

    def forward(self, x):
        x = x.view(-1, 2*self.n_joints)
        x = self.enc_inp_block(x)

        ''' Similar to Hands VAE'''
        x = self.LA_block(x)
        x = self.LA_block(x)
        x = self.LA_block(x)
        x = self.LA_block(x)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class Decoder3D(nn.Module):
    def __init__(self, latent_dim, n_joints=16, activation=nn.ReLU):
        super(Decoder3D, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_joints = n_joints
        self.neurons = 512
        self.name = "Decoder3D"
        self.drop_out_p = 0

        self.__build_model()

    def __build_model(self):

        self.dec_inp_block = nn.Sequential(
            nn.Linear(self.latent_dim, self.neurons),
            nn.BatchNorm1d(self.neurons),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )

        self.dec_out_block = nn.Sequential(
            # TODO is it good idea to have activation \
            # and drop out at the end for enc or dec
            nn.Linear(self.neurons, 3*self.n_joints),
        )

        self.LBAD_block = nn.Sequential(
            nn.Linear(self.neurons, self.neurons),
            nn.BatchNorm1d(self.neurons),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )

        self.LA_block = nn.Sequential(
            nn.Linear(self.neurons, self.neurons),
            self.activation()
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        x = self.dec_inp_block(x)

        x = self.LA_block(x)
        x = self.LA_block(x) 
        x = self.LA_block(x)
        x = self.LA_block(x) 

        x = self.dec_out_block(x)

        return x


def reparameterize(mean, logvar, eval=False):
    # TODO not sure why few repos do that
    if eval:
        return mean

    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)

    return mean + eps*std


def PJPE(pred, target):
    '''
    Equation per sample per sample in batch
    PJPE(per joint position estimation) -- root((x-x`)2+(y-y`)2+(z-z`)2)

    Arguments:
        pred (tensor)-- predicted 3d poses [n,j,3]
        target (tensor)-- taget 3d poses [n,j,3]
    Returns:
        PJPE -- calc MPJPE - mean(PJPE, axis=0) for each joint across batch
    '''
    PJPE = torch.sqrt(
        torch.sum((pred-target).pow(2), dim=2))

    # MPJPE = torch.mean(PJPE, dim=0)

    return PJPE


def KLD(mean, logvar, decoder_name):
    '''
    Returns:
        loss -- averaged with the same denom as of recon
    '''
    loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    if '3D' in decoder_name:
        # normalize by same number in recon - b*j*dim
        # from vae-hands local_utility_fn ln-108
        loss /= mean.shape[0]*16*3
    elif 'RGB' in decoder_name:
        print("[WARNING] fix KLD loss normalization for current decoder")
        loss /= mean.shape[0]*256*256
    else:
        print(f"[WARNING] {decoder_name} has no KLD loss normalization")

    return loss


'''
test function for sanity check only - ignore
'''


def test(inp, target):
    latent_dim = 2

    # inp = torch.randn(2, 16, 2)  # [b, joints. 2]
    # target = torch.randn(2, 16, 3)  # [b, joints, 3]

    encoder_2d = Encoder2D(latent_dim)
    decoder_3d = Decoder3D(latent_dim)
    encoder_2d.eval()
    decoder_3d.eval()

    # print(encoder_2d.parameters)

    with torch.no_grad():

        # 2D -> 3D
        mean, logvar = encoder_2d(inp)
        z = reparameterize(mean, logvar)
        pose3d = decoder_3d(z)
        pose3d = pose3d.view(-1, 16, 3)
        recon_loss = PJPE(pose3d, target)
        # loss = nn.functional.l1_loss(pose3d, target)
        kld_loss = KLD(mean, logvar, decoder_3d.__class__.__name__)
        print("2D -> 3D", recon_loss)


if __name__ == "__main__":

    pose2d = [[-0.0520,  0.5179],
              [-0.3927, -1.1830],
              [0.0502, -1.4042],
              [0.0635, -0.2084],
              [0.2380, -1.2675],
              [0.3716, -1.4058],
              [0.2500,  0.5317],
              [0.0531,  0.6651],
              [0.1978,  0.6016],
              [0.1708,  0.5921],
              [0.1230,  0.8581],
              [0.6800,  1.1669],
              [1.3780,  0.6804],
              [-0.0196,  0.7375],
              [-0.3905,  1.2061],
              [-0.5927,  0.7949]]

    pose3d = [[-0.0915,  1.4068,  0.9755],
              [-1.4016, -0.9308,  1.3867],
              [1.4055, -0.9582,  1.2904],
              [0.0914, -1.4068, -0.9755],
              [1.2805, -0.9814,  1.3715],
              [1.3662, -0.8682,  1.3793],
              [1.2664,  1.1126,  0.3048],
              [0.7281,  0.4539,  0.8268],
              [1.2756, -0.9743,  0.7223],
              [0.9169, -0.1797, -0.4335],
              [0.8026,  1.4129,  1.4049],
              [0.9368,  0.6461,  1.3693],
              [0.7160,  0.4598,  1.2660],
              [0.8033,  1.3638,  0.7996],
              [-0.9785,  0.6049,  0.6722],
              [-0.9765,  0.5890,  0.8608]]

    test(torch.FloatTensor([pose2d]), torch.FloatTensor([pose3d]))