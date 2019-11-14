from data import DataManager

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import tensor

from src.Config import Config
from typing import Tuple


def load_data(batch_size=Config.batch_size, dataset=""):
    """loads a dataset using the torch dataloader and and the settings in Config"""

    data_loader_args = {'num_workers': Config.num_workers, 'pin_memory': False if Config.device != 'cpu' else False}
    data_path = DataManager.get_datasets_folder()

    colour_image_transform = transforms.Compose([
        #image transform goes here
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    black_and_white_image_transform = transforms.Compose([
        #image transform goes here
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    download = False

    if dataset == "":
        dataset = Config.dataset.lower()

    if dataset == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST(data_path,
                           train=True,
                           download=download,
                           transform=black_and_white_image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args)

        test_loader = DataLoader(
            datasets.MNIST(data_path,
                           train=False,
                           download=download,
                           transform=black_and_white_image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args)

    elif dataset == 'fashion_mnist':
        train_loader = DataLoader(
            datasets.FashionMNIST(data_path,
                                  train=True, download=download,
                                  transform=black_and_white_image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )

        test_loader = DataLoader(
            datasets.FashionMNIST(data_path,
                                  train=False, download=download,
                                  transform=black_and_white_image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )
    elif dataset == 'cifar10':
        train_loader = DataLoader(
            datasets.CIFAR10(data_path,
                             train=True, download=download,
                             transform=colour_image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )

        test_loader = DataLoader(
            datasets.CIFAR10(data_path,
                             train=False, download=download,
                             transform=colour_image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )

    else:
        raise Exception('Invalid dataset name, options are fashion_mnist, mnist or cifar10 you provided:' + dataset)

    return train_loader, test_loader


def sample_data(device, batch_size=Config.batch_size, dataset=Config.dataset) -> Tuple[tensor, tensor]:
    """returns a single batch of the dataset named in Config"""
    train_loader, test_loader = load_data(batch_size=batch_size, dataset=dataset)
    input, target = next(iter(train_loader))
    return input.to(device), target.to(device)

    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    #     return inputs.to(device), targets.to(device)
