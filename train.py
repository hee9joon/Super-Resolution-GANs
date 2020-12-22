import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from dataset import get_div2k_loader
from models import Discriminator, Generator_SRGAN, Generator_ESRGAN
from losses import PerceptualLoss, TVLoss
from inference import inference
from utils import make_dirs, get_lr_scheduler, set_requires_grad, sample_images

# Reproducibility #
cudnn.deterministic = True
cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():

    # Fix Seed for Reproducibility #
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(9)

    # Samples, Weights and Results Path #
    paths = [config.samples_path, config.weights_path]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    train_div2k_loader = get_div2k_loader(sort='train',
                                          batch_size=config.batch_size,
                                          image_size=config.image_size,
                                          upscale_factor=config.upscale_factor,
                                          crop_size=config.crop_size)
    val_div2k_loader = get_div2k_loader(sort='val',
                                        batch_size=config.val_batch_size,
                                        image_size=config.image_size,
                                        upscale_factor=config.upscale_factor,
                                        crop_size=config.crop_size)
    total_batch = len(train_div2k_loader)

    # Prepare Networks #
    D = Discriminator()

    if config.sort == 'SRGAN':
        G = Generator_SRGAN()
    elif config.sort == 'ESRGAN':
        G = Generator_ESRGAN()
    else:
        raise NotImplementedError

    networks = [D, G]
    for network in networks:
        network.to(device)

    # Loss Function #
    criterion_Perceptual = PerceptualLoss(sort=config.sort).to(device)

    # For SRGAN #
    criterion_MSE = nn.MSELoss()
    criterion_TV = TVLoss()

    # For ESRGAN #
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_Content = nn.L1Loss()

    # Optimizers #
    D_optim = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.9, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.9, 0.999))

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_losses, G_losses = [], []

    # Train #
    print("Training {} started with total epoch of {}.".format(config.sort, config.num_epochs))

    for epoch in range(config.num_epochs):
        for i, (high, low) in enumerate(train_div2k_loader):

            D.train()
            if config.sort == "SRGAN":
                G.train()

            # Data Preparation #
            high = high.to(device)
            low = low.to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            set_requires_grad(D, requires_grad=True)

            # Generate Fake HR Images #
            fake_high = G(low)

            if config.sort == 'SRGAN':

                # Forward Data #
                prob_real = D(high)
                prob_fake = D(fake_high.detach())

                # Calculate Total Discriminator Loss #
                D_loss = 1 - prob_real.mean() + prob_fake.mean()

            elif config.sort == 'ESRGAN':

                # Forward Data #
                prob_real = D(high)
                prob_fake = D(fake_high.detach())

                # Relativistic Discriminator #
                diff_r2f = prob_real - prob_fake.mean()
                diff_f2r = prob_fake - prob_real.mean()

                # Labels #
                real_labels = torch.ones(diff_r2f.size()).to(device)
                fake_labels = torch.zeros(diff_f2r.size()).to(device)

                # Adversarial Loss #
                D_loss_real = criterion_BCE(diff_r2f, real_labels)
                D_loss_fake = criterion_BCE(diff_f2r, fake_labels)

                # Calculate Total Discriminator Loss #
                D_loss = (D_loss_real + D_loss_fake).mean()

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            ###################
            # Train Generator #
            ###################

            set_requires_grad(D, requires_grad=False)

            if config.sort == 'SRGAN':

                # Adversarial Loss #
                prob_fake = D(fake_high).mean()
                G_loss_adversarial = torch.mean(1 - prob_fake)
                G_loss_mse = criterion_MSE(fake_high, high)

                # Perceptual Loss #
                lambda_perceptual = 6e-3
                G_loss_perceptual = criterion_Perceptual(fake_high, high)

                # Total Variation Loss #
                G_loss_tv = criterion_TV(fake_high)

                # Calculate Total Generator Loss #
                G_loss = config.lambda_adversarial * G_loss_adversarial + G_loss_mse + lambda_perceptual * G_loss_perceptual + config.lambda_tv * G_loss_tv

            elif config.sort == 'ESRGAN':

                # Forward Data #
                prob_real = D(high)
                prob_fake = D(fake_high)

                # Relativistic Discriminator #
                diff_r2f = prob_real - prob_fake.mean()
                diff_f2r = prob_fake - prob_real.mean()

                # Labels #
                real_labels = torch.ones(diff_r2f.size()).to(device)
                fake_labels = torch.zeros(diff_f2r.size()).to(device)

                # Adversarial Loss #
                G_loss_bce_real = criterion_BCE(diff_f2r, real_labels)
                G_loss_bce_fake = criterion_BCE(diff_r2f, fake_labels)

                G_loss_bce = (G_loss_bce_real + G_loss_bce_fake).mean()

                # Perceptual Loss #
                lambda_perceptual = 1e-2
                G_loss_perceptual = criterion_Perceptual(fake_high, high)

                # Content Loss #
                G_loss_content = criterion_Content(fake_high, high)

                # Calculate Total Generator Loss #
                G_loss = config.lambda_bce * G_loss_bce + lambda_perceptual * G_loss_perceptual + config.lambda_content * G_loss_content

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("{} | Epoch [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(config.sort, epoch + 1, config.num_epochs, i + 1, total_batch, np.average(D_losses), np.average(G_losses)))

                # Save Sample Images #
                sample_images(val_div2k_loader, G, epoch, config.samples_path)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights and Inference #
        if (epoch+1) % config.save_every == 0:
            torch.save(G.state_dict(), os.path.join(config.weights_path, '{}_Generator_Epoch_{}.pkl'.format(config.sort, epoch + 1)))
            inference(G, val_div2k_loader, epoch, config.inference_path)

    print("Training Finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()