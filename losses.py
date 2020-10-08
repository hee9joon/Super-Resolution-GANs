import torch
import torch.nn as nn
import torchvision.models as models


class TVLoss(nn.Module):
    """Total Variation Loss for Super Resolution GAN"""
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        height_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :height - 1, :]), 2).sum()
        width_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :width - 1]), 2).sum()
        out = 2 * (height_tv / count_h + width_tv / count_w) / batch_size
        out = self.tv_loss_weight * out
        return out

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class PerceptualLoss(nn.Module):
    """Perception Loss using VGG19 for Enhanced Super Resolution GAN"""
    def __init__(self, sort):
        super(PerceptualLoss, self).__init__()

        self.sort = sort
        vgg = models.vgg19(pretrained=True)
        if self.sort == "SRGAN":
            relu = 31
        elif self.sort == "ESRGAN":
            relu = 35
        else:
            raise NotImplementedError
        model = nn.Sequential(*list(vgg.features)[:relu])
        model = model.eval()

        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, source, target):
        source_feature = self.vgg(source)
        target_feature = self.vgg(target)
        if self.sort == "SRGAN":
            perception_loss = self.mse_loss(source_feature, target_feature)
        elif self.sort == "ESRGAN":
            perception_loss = self.mae_loss(source_feature, target_feature)
        else:
            raise NotImplementedError

        return perception_loss