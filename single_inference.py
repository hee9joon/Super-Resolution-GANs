import os
import numpy as np

import torch
from torchvision.utils import save_image

from config import *
from dataset import get_div2k_loader
from models import Generator_SRGAN, Generator_ESRGAN
from utils import denorm

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def single_inference():

    results_path = './results/single/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Configure the Network #
    SRGAN = Generator_SRGAN().to(device)
    ESRGAN = Generator_ESRGAN().to(device)

    # Configuration of Epochs and Paths #
    srgan_weight_path = os.path.join(config.weights_path, 'SRGAN_Generator_Epoch_{}.pkl'.format(config.num_epochs))
    esrgan_weight_path = os.path.join(config.weights_path, 'ESRGAN_Generator_Epoch_{}.pkl'.format(config.num_epochs))

    # Prepare Generator #
    SRGAN.load_state_dict(torch.load(srgan_weight_path))
    ESRGAN.load_state_dict(torch.load(esrgan_weight_path))

    SRGAN.eval()

    # Prepare Data Loader #
    div2k_loader = get_div2k_loader(sort='val',
                                    batch_size=config.val_batch_size,
                                    image_size=config.image_size,
                                    upscale_factor=config.upscale_factor,
                                    crop_size=config.crop_size)

    # Up-sampling Network #
    up_sampler = torch.nn.Upsample(scale_factor=config.upscale_factor, mode='bicubic').to(device)

    for i, (high, low) in enumerate(div2k_loader):

        # Prepare Data #
        high = high.to(device)
        low = low.to(device)

        # Forward Data to Networks #
        with torch.no_grad():
            bicubic = up_sampler(low.detach())
            fake_high_sr = SRGAN(low.detach())
            fake_high_esr = ESRGAN(low.detach())

        # Normalize and Save Images #
        save_image(denorm(bicubic.data),
                   os.path.join(results_path, 'Inference_Samples_BICUBIC_%03d.png' % (i+1)))

        save_image(denorm(fake_high_sr.data),
                   os.path.join(results_path, 'Inference_Samples_SRGAN_%03d.png' % (i+1)))

        save_image(denorm(fake_high_esr.data),
                   os.path.join(results_path, 'Inference_Samples_ESRGAN_%03d.png' % (i+1)))

        save_image(denorm(high.data),
                   os.path.join(results_path, 'Inference_Samples_TARGET_%03d.png' % (i+1)))


if __name__ == "__main__":
    single_inference()