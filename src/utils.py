import torch
import torch.nn.functional as F


def kaiming_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")

    # classname = m.__class__.__name__
    # # if m.name == 'Critic':
    # if classname.find('Linear') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    # else:
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


def PJPE(pred, target):
    """
    Equation per sample per sample in batch
    PJPE(per joint position estimation) -- root((x-x`)2+(y-y`)2+(z-z`)2)

    Arguments:
        pred (tensor)-- predicted 3d poses [n,j,3]
        target (tensor)-- taget 3d poses [n,j,3]
    Returns:
        PJPE -- calc MPJPE - mean(PJPE, axis=0) for each joint across batch
    """
    PJPE = torch.sqrt(torch.sum((pred - target).pow(2), dim=2))
    # MPJPE = torch.mean(PJPE, dim=0)
    return PJPE


def KLD(mean, logvar):
    """
    Returns:
        loss -- averaged with the same denom as of recon
    """

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0
    )
    return kld_loss

    # # TODO
    # loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    # # normalize by same number in recon - b*j*dim
    # # from vae-hands local_utility_fn ln-108
    # loss /= mean.shape[0] * 16 * 3
