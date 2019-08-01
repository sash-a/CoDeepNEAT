from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from data import DataManager
from src.Config import Config
import os


def load_data(batch_size=64, dataset=""):
    data_loader_args = {'num_workers': Config.num_workers, 'pin_memory': False if Config.device != 'cpu' else False}
    data_path = DataManager.get_Datasets_folder()

    colour_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    black_and_white_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    download = False

    if dataset == "":
        dataset = Config.dataset.lower()

    # print("loading(", dataset, ")data from:", data_path)

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


def sample_data(device, batch_size=16, dataset=""):
    train_loader, test_loader = load_data(batch_size=batch_size, dataset=dataset)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        return inputs.to(device), targets.to(device)
