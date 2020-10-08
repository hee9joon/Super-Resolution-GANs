import os
import numpy as np

import torch
from torchvision.utils import save_image

from config import *
from utils import PSNR, MSE, SSIM, denorm

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference(generator, data_loader, epoch, path):
    """Calculate metric of GT and Generated Images and Save Results"""

    # Inference Path #
    results_path = os.path.join(path, '{}_Epoch_{}'.format(config.sort, epoch+1))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Lists #
    PSNR_GT_values, PSNR_Gen_values = [], []
    MSE_GT_values, MSE_Gen_values = [], []
    SSIM_GT_values, SSIM_Gen_values = [], []

    # Up-sampling Network #
    up_sampler = torch.nn.Upsample(scale_factor=config.upscale_factor, mode='bicubic').to(device)

    # Generator #
    if config.sort == "SRGAN":
        generator.eval()

    ############################################################################
    # Calculate Metric for Ground Truths and Generated Metric and Save Results #
    ############################################################################
    print("Inference Results at Epoch {} follows:".format(epoch+1))
    for i, (high, low) in enumerate(data_loader):

        # Prepare Data #
        high = high.to(device)
        low = low.to(device)

        # Forward Data to Networks #
        with torch.no_grad():
            bicubic = up_sampler(low.detach())
            fake_high = generator(low.detach())

        # Calculate PSNR #
        PSNR_GT_value = PSNR(high.detach().cpu().numpy(), bicubic.detach().cpu().numpy())
        PSNR_Gen_value = PSNR(high.detach().cpu().numpy(), fake_high.detach().cpu().numpy())

        # Calculate MSE #
        MSE_GT_value = MSE(high.detach().cpu().numpy(), bicubic.detach().cpu().numpy())
        MSE_Gen_value = MSE(high.detach().cpu().numpy(), fake_high.detach().cpu().numpy())

        # Calculate SSIM #
        SSIM_GT_value = SSIM(high.detach().cpu().numpy(), bicubic.detach().cpu().numpy())
        SSIM_Gen_value = SSIM(high.detach().cpu().numpy(), fake_high.detach().cpu().numpy())

        # Add items to Lists #
        PSNR_GT_values.append(PSNR_GT_value)
        PSNR_Gen_values.append(PSNR_Gen_value)

        MSE_GT_values.append(MSE_GT_value)
        MSE_Gen_values.append(MSE_Gen_value)

        SSIM_GT_values.append(SSIM_GT_value)
        SSIM_Gen_values.append(SSIM_Gen_value)

        # Normalize and Save Images #
        images = [bicubic, fake_high, high]

        result = torch.cat(images, dim=0)

        save_image(denorm(result.data),
                   os.path.join(results_path, '%s_Inference_Epoch_%03d_Samples_%03d.png' % (config.sort, epoch + 1, i+1)),
                   nrow=len(images))

    ####################
    # Print Statistics #
    ####################

    print("PSNR Bicubic | Average {:.3f} | S.D {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.mean(PSNR_GT_values), np.std(PSNR_GT_values), np.max(PSNR_GT_values), np.min(PSNR_GT_values)))
    print("PSNR Generated at Epoch {} | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(epoch + 1, np.average(PSNR_Gen_values), np.std(PSNR_Gen_values), np.max(PSNR_Gen_values),
                  np.min(PSNR_Gen_values)))

    print("MSE Bicubic | Average {:.3f} | S.D {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.mean(MSE_GT_values), np.std(MSE_GT_values), np.max(MSE_GT_values), np.min(MSE_GT_values)))
    print("MSE Generated at Epoch {} | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(epoch + 1, np.mean(MSE_Gen_values), np.std(MSE_Gen_values), np.max(MSE_Gen_values),
                  np.min(MSE_Gen_values)))

    print("SSIM Bicubic | Average {:.3f} | S.D {:.3f} | Maximum {:.3f} | Minimum {:.3f}"
          .format(np.mean(SSIM_GT_values), np.std(SSIM_GT_values), np.max(SSIM_GT_values), np.min(SSIM_GT_values)))
    print("SSIM Generated at Epoch {} | Average {:.3f} | SD {:.3f} | Maximum {:.3f} | Minimum {:.3f}\n"
          .format(epoch + 1, np.mean(SSIM_Gen_values), np.std(SSIM_Gen_values), np.max(SSIM_Gen_values),
                  np.min(SSIM_Gen_values)))