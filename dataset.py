import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DIV2K(Dataset):
    def __init__(self, sort, image_size, crop_size, upscale_factor):
        super(DIV2K, self).__init__()

        self.sort = sort
        self.train_hr_path = './data/train_hr/'
        self.train_lr_path = './data/train_lr/'

        self.val_hr_path = './data/val_hr/'
        self.val_lr_path = './data/val_lr/'

        self.image_size = image_size
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

        self.train_images = [x for x in sorted(os.listdir(self.train_hr_path))]
        self.val_images = [x for x in sorted(os.listdir(self.val_hr_path))]

        self.high_res = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.low_res = transforms.Compose([
            transforms.Resize((self.image_size // upscale_factor, self.image_size // upscale_factor)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        if self.sort == 'train':
            train_hr_image = Image.open(os.path.join(self.train_hr_path, self.train_images[index])).convert("RGB")
            train_lr_image = Image.open(os.path.join(self.train_lr_path, self.train_images[index])).convert("RGB")

            high_image = self.high_res(train_hr_image)
            low_image = self.low_res(train_lr_image)

        elif self.sort == 'val':
            val_hr_image = Image.open(os.path.join(self.val_hr_path, self.val_images[index])).convert("RGB")
            val_lr_image = Image.open(os.path.join(self.val_lr_path, self.val_images[index])).convert("RGB")

            high_image = self.high_res(val_hr_image)
            low_image = self.low_res(val_lr_image)

        else:
            raise NotImplementedError

        return high_image, low_image

    def __len__(self):
        return len(self.train_images) if self.sort == 'train' else len(self.val_images)


def get_div2k_loader(sort, batch_size, image_size, upscale_factor, crop_size):
    if sort == 'train':
        dataset = DIV2K(sort, image_size, crop_size, upscale_factor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif sort == 'val':
        dataset = DIV2K(sort, image_size, crop_size, upscale_factor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    else:
        raise NotImplementedError

    return data_loader