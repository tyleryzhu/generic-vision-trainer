from __future__ import annotations

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
import h5py

ImageFile.LOAD_TRUNCATED_IMAGES = True


# from torchvision.transforms import Compose, Normalize, PILToTensor, Resize


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, color_jitter=0.4):
        self.train = train

        self.transform = self._get_transforms(color_jitter=color_jitter)
        self.dataset = CIFAR10(
            root="/data/cifar10", train=self.train, download=True
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _get_transforms(self, color_jitter=0.4):
        if self.train:
            transform = [
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        else:
            transform = [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        return transforms.Compose(transform)
