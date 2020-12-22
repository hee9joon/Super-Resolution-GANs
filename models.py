import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d

import math

from config import *


class Discriminator(nn.Module):
    """Discriminator Network for Super Resolution GAN and Enhanced One"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.ndf = 64
        self.linear_dim = 1000
        self.out_channels = 1

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 2, self.ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 4, self.ndf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 8, self.ndf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fcn = nn.Sequential(
            nn.Linear(self.ndf * 8, self.linear_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.linear_dim, self.out_channels),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.ndf * 8, self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.patch = nn.Sequential(
            nn.Conv2d(self.ndf * 8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)

        if config.disc_type == 'fcn':
            out = adaptive_avg_pool2d(input=out, output_size=1)
            out = torch.flatten(out, 1)
            out = self.fcn(out)

        elif config.disc_type == 'conv':
            out = adaptive_avg_pool2d(input=out, output_size=1)
            out = self.conv(out)

        elif config.disc_type == 'patch':
            out = self.patch(out)

        return out


class Generator_SRGAN(nn.Module):
    """Generator Network for Super Resolution GAN"""
    def __init__(self, scale_factor=4):
        super(Generator_SRGAN, self).__init__()

        self.in_channels = 3
        self.ngf = 64
        self.out_channels = 3

        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ngf, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.block2 = nn.Sequential(
            ResidualBlock(self.ngf),
            ResidualBlock(self.ngf),
            ResidualBlock(self.ngf),
            ResidualBlock(self.ngf),
            ResidualBlock(self.ngf)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ngf)
        )

        num_upsampling = int(math.log(scale_factor, 2))
        block4 = list()

        for i in range(num_upsampling):
            block4 += [
                Upsample(self.ngf)
            ]

        self.block4 = nn.Sequential(*block4)

        self.block5 = nn.Sequential(
            nn.Conv2d(self.ngf, self.out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = out3 + out1
        out5 = self.block4(out4)
        out6 = self.block5(out5)
        return out6


class ResidualBlock(nn.Module):
    """Residual Block for Super Resolution GAN"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        out = self.main(x)
        out += x
        return out


class Upsample(nn.Module):
    """Upsampling for Super Resolution GAN and Enhanced One"""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, activation='PReLU'):
        super(Upsample, self).__init__()

        activation = nn.PReLU() if activation == 'PReLU' else nn.ReLU()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PixelShuffle(upscale_factor=2),
            activation
        )

    def forward(self, x):
        out = self.main(x)
        return out


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for Enhanced Super Resolution GAN"""
    def __init__(self):
        super(ResidualDenseBlock, self).__init__()

        self.dim = 64
        self.growth_rate = 32
        self.scale_ration = 0.2

        self.block1 = nn.Sequential(
            nn.Conv2d(self.dim, self.growth_rate, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.dim + self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.dim + 2 * self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(self.dim + 3 * self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(self.dim + 4 * self.growth_rate, self.dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(torch.cat((out1, x), dim=1))
        out3 = self.block3(torch.cat((out1, out2, x), dim=1))
        out4 = self.block4(torch.cat((out1, out2, out3, x), dim=1))
        out5 = self.block5(torch.cat((out1, out2, out3, out4, x), dim=1))
        out5 *= self.scale_ration
        out5 = out5 + x
        return out5


class ResidualInResidualDenseBlock(nn.Module):
    """Residual In Residual Dense Block for Enhanced Super Resolution GAN"""
    def __init__(self):
        super(ResidualInResidualDenseBlock, self).__init__()

        self.scale_ration = 0.2

        self.main = nn.Sequential(
            ResidualDenseBlock(),
            ResidualDenseBlock(),
            ResidualDenseBlock()
        )

    def forward(self, x):
        out = self.main(x)
        out *= self.scale_ration
        out += x
        return out


class Generator_ESRGAN(nn.Module):
    """Generator Network for Enhanced Super Resolution GAN"""
    def __init__(self, scale_factor=4):
        super(Generator_ESRGAN, self).__init__()

        self.in_channels = 3
        self.ngf = 64
        self.num_blocks = 23
        self.out_channels = 3

        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.in_channels, self.ngf, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        blocks = list()
        for i in range(self.num_blocks):
            blocks += [
                ResidualInResidualDenseBlock()
            ]

        self.block2 = nn.Sequential(*blocks)

        self.block3 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        upsampling = list()
        num_upsampling = int(math.log(scale_factor, 2))

        for j in range(num_upsampling):
            upsampling += [
                Upsample(self.ngf, kernel_size=1, stride=1, padding=0, activation='ReLU')
            ]

        self.block4 = nn.Sequential(*upsampling)

        self.block5 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.block6 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.ngf, self.out_channels, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3 + out1)
        out5 = self.block5(out4)
        out = self.block6(out5)
        return out