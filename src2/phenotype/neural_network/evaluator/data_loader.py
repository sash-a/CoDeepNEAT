from os import path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

from data import DataManager

from src2.configuration import config
from src2.phenotype.augmentations.augmentation_scheme import AugmentationScheme


def load_data(composed_transforms: transforms.Compose, split: str) -> DataLoader:
    """
    Loads the data given the config.dataset

    :param composed_transforms: the augmentations to apply to the dataset
    :param split: either train or test, dataset returned also depends on config.fully_train -
    'train' will return 42500 images for evolution, but 50000 for fully training.
    'test' will return 7500 images for evolution, but 10000 for fully training.

    Note: the validation/train split does not try balance the classes of data, it just takes the first n for the train
    set and the remaining data goes to the validation set
    """
    if split.lower() not in ['train', 'test']:
        raise ValueError('Parameter split can be one of train, test or validation, but received: ' + str(split))

    # when to load train set
    train: bool = True if split == 'train' or (split == 'test' and not config.fully_train) else False

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

    if train and not config.fully_train:
        # Splitting the train set into a train and valid set
        train_size = int(len(dataset) * (1 - config.validation_split))
        if split == 'train':
            dataset = Subset(dataset, range(train_size))
        else:
            dataset = Subset(dataset, range(train_size, len(dataset)))

    print(split, 'set size in', 'FT' if config.fully_train else 'evo', len(dataset))

    # TODO: test num workers and pin memory
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)


def get_data_shape() -> List[int]:
    return list(next(iter(load_data(load_transform(), 'test')))[0].size())


def get_generic_dataset(composed_transforms: transforms.Compose, train: bool) -> Dataset:
    """
    Loads data from custom_dataset_root, given that the data is structured as follows:
    root/train/dog/xxx.png
    root/test/dog/xxy.png

    root/train/cat/123.png
    root/test/cat/nsdf3.png

    Note: the train set should contain enough data to be split into a train and validation set
    :return: a train and test dataloader
    """
    if train:
        data = ImageFolder(root=path.join(config.custom_dataset_root, 'train'), transform=composed_transforms)
    else:
        data = ImageFolder(root=path.join(config.custom_dataset_root, 'test'), transform=composed_transforms)

    return data


def load_transform(aug: AugmentationScheme = None) -> transforms.Compose:
    if config.evolve_da and not config.batch_augmentation and aug is not None:
        return transforms.Compose([
            aug,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
