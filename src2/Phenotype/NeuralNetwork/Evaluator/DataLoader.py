from typing import Tuple

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from os import path

from src2.Configuration import config


def load_data(composed_transforms: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
    """Loads the data given the config.dataset"""
    pass


def generic_dataloader(composed_transforms: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
    """
    Loads data from custom_dataset_root, given that the data is structured as follows:
    root/train/dog/xxx.png
    root/test/dog/xxy.png

    root/train/cat/123.png
    root/test/cat/nsdf3.png
    :return: a train and test dataloader
    """
    train_data = ImageFolder(root=path.join(config.custom_dataset_root, 'train'), transform=composed_transforms)
    test_data = ImageFolder(root=path.join(config.custom_dataset_root, 'test'), transform=composed_transforms)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)

    return train_loader, test_loader
