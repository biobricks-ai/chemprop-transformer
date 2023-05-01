import torch.nn.functional as F
import torch

def vae_loss(x_output, x, z_mean, z_logvar):
    recon_loss = F.binary_cross_entropy(x_output, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss