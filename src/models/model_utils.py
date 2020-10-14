
import torch 


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # classname = m.__class__.__name__
    # # if m.name == 'Critic':    
    # if classname.find('Linear') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    # else:
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    

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
    torch.nn.L1Loss()
    return PJPE

# TODO add static methods wherever neccesary
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
        print(f"[WARNING] {decoder_name} has no KLD loss normalization implemented")

    return loss

@staticmethod
def calculate_rotation(xy_real, z_pred):
    xy_split = F.split_axis(xy_real, xy_real.data.shape[1], axis=1)
    z_split = F.split_axis(z_pred, z_pred.data.shape[1], axis=1)
    # Vector v0 (neck -> nose) on zx-plain. v0=(a0, b0).
    a0 = z_split[9] - z_split[8]
    b0 = xy_split[9 * 2] - xy_split[8 * 2]
    n0 = F.sqrt(a0 * a0 + b0 * b0)
    # Vector v1 (right shoulder -> left shoulder) on zx-plain. v1=(a1, b1).
    a1 = z_split[14] - z_split[11]
    b1 = xy_split[14 * 2] - xy_split[11 * 2]
    n1 = F.sqrt(a1 * a1 + b1 * b1)

    # Return sine value of the angle between v0 and v1.
    return (a0 * b1 - a1 * b0) / (n0 * n1)


@staticmethod
def calculate_heuristic_loss(xy_real, z_pred):
    return F.average(F.relu(
        calculate_rotation(xy_real, z_pred)))


if __name__ == "__main__":
    print("[INFO] Method for Models")