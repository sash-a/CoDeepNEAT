import random
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from os import path

from data import DataManager
from src2.Configuration import config

import matplotlib.pyplot as plt
import numpy as np


def load_data(composed_transforms: transforms.Compose, split: str) -> DataLoader:
    """Loads the data given the config.dataset"""
    if split not in ['train', 'test', 'validation']:
        raise ValueError('Parameter split can be one of train, test or validation, but received: ' + str(split))

    train: bool = True if split == 'train' or split == 'validation' else False

    dataset_args = {
        'root': DataManager.get_datasets_folder(),
        'train': train,
        'download': config.download_dataset,
        'transform': composed_transforms
    }

    if config.dataset == 'mnist':
        dataset = MNIST(**dataset_args)
    elif config.dataset == 'cifar10':
        dataset = CIFAR10(**dataset_args)
    elif config.dataset == 'custom':
        dataset = get_generic_dataset(composed_transforms, train)
    else:
        raise ValueError('config.dataset can be one of mnist, cifar10 or custom, but received: ' + str(config.dataset))

    if train:
        # Splitting the train set into a train and valid set
        train_size = int(len(dataset) * (1 - config.validation_split))
        validation_size = len(dataset) - train_size
        train, valid = random_split(dataset, [train_size, validation_size])

        if split == 'train':
            dataset = train
        else:
            dataset = valid
    # TODO: test num workers and pin memory

    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)


def get_data_shape() -> List[int]:
    composed_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return list(next(iter(load_data(composed_transform, 'test')))[0].size())


def get_generic_dataset(composed_transforms: transforms.Compose, train: bool) -> Dataset:
    """
    Loads data from custom_dataset_root, given that the data is structured as follows:
    root/train/dog/xxx.png
    root/test/dog/xxy.png

    root/train/cat/123.png
    root/test/cat/nsdf3.png
    :return: a train and test dataloader
    """
    if train:
        data = ImageFolder(root=path.join(config.custom_dataset_root, 'train'), transform=composed_transforms)
    else:
        data = ImageFolder(root=path.join(config.custom_dataset_root, 'test'), transform=composed_transforms)

    return data


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
