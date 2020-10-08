import os
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity

import torch
from torch.nn import init
from torchvision.utils import save_image

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def init_weights_normal(m):
    """Normal Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)


def init_weights_xavier(m):
    """Xavier Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)


def init_weights_kaiming(m):
    """Kaiming He Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


def get_lr_scheduler(optimizer):
    """Learning Rate Scheduler"""
    if config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay_rate)
    elif config.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler


def set_requires_grad(network, requires_grad=False):
    """Prevent a Network from Updating"""
    for param in network.parameters():
        param.requires_grad = requires_grad


def denorm(x):
    """De-normalization"""
    out = (x+1) / 2
    return out.clamp(0, 1)


def PSNR(image_A, image_B):
    """Calculate PSNR Value for a Pair of Images"""
    image_A = np.array(image_A, dtype=np.float32)
    image_B = np.array(image_B, dtype=np.float32)
    mse = np.mean((image_A - image_B) ** 2)
    max_pixel = image_B.max() - image_B.min()
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def MSE(image_A, image_B):
    """Calculate MSE Value for a Pair of Images"""
    image_A = np.array(image_A, dtype=np.float32)
    image_B = np.array(image_B, dtype=np.float32)
    mse = np.sum((image_A - image_B) ** 2)
    mse /= float((image_A.shape[2] * image_A.shape[3]))
    return mse


def SSIM(image_A, image_B):
    """Calculate SSIM Value for a Pair of Images"""
    image_A = np.array(image_A, dtype=np.float32)
    image_B = np.array(image_B, dtype=np.float32)
    image_A = np.transpose(image_A, (0, 2, 3, 1)).squeeze(axis=0)
    image_B = np.transpose(image_B, (0, 2, 3, 1)).squeeze(axis=0)
    ssim = structural_similarity(image_A, image_B, data_range=image_B.max() - image_B.min(), multichannel=True)
    return ssim


def sample_images(data_loader, generator, epoch, path):
    """Save Sample Images for Every Epoch"""

    high, low = next(iter(data_loader))

    low = low.to(device)
    high = high.to(device)

    if config.sort == "SRGAN":
        generator.eval()

    up_sampler = torch.nn.Upsample(scale_factor=config.upscale_factor, mode='bicubic').to(device)
    bicubic = up_sampler(low)

    with torch.no_grad():
        fake_high = generator(low.detach())

    images = [bicubic, fake_high, high]

    result = torch.cat(images, dim=0)
    save_image(denorm(result.data),
               os.path.join(path, '%s_Samples_Epoch_%03d.png' % (config.sort, epoch + 1)),
               nrow=8 if config.sort == "SRGAN" else len(images))

    del images