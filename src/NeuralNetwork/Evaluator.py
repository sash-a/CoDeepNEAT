# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.DataAugmentation import BatchAugmentor
from src.Config import Config
from data import DataManager

import time
import torch.multiprocessing as mp

printBatchEvery = -1  # -1 to switch off batch printing
print_epoch_every = 1


def train(model, train_loader, epoch, test_loader, device, augmentor=None, print_accuracy=False):
    """
    Run a single train epoch

    :param model: the network of type torch.nn.Module
    :param train_loader: the training dataset
    :param epoch: the current epoch
    :param test_loader: The test dataset loader
    :param print_accuracy: True if should test when printing batch info
    """
    print('Train received device:', device)
    model.train()

    loss = 0
    batch_idx = 0

    s = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if augmentor is not None:
            aug_inputs, aug_labels = BatchAugmentor.augment_batch(inputs.numpy(), targets.numpy(), augmentor)

        if Config.interleaving_check:
            print('in train', mp.current_process().name)
            sys.stdout.flush()

        inputs, targets = inputs.to(device), targets.to(device)

        model.optimizer.zero_grad()

        output = model(inputs)
        m_loss = model.loss_fn(output, targets.float())
        del inputs
        del targets
        # augmented_inputs, augmented_labels = augmented_inputs.to(device), augmented_labels.to(device)
        m_loss.backward()
        model.optimizer.step()

        loss += m_loss.item()

        if augmentor is not None:
            aug_inputs, aug_labels = aug_inputs.to(device), aug_labels.to(device)
            # print("training on augmented images shape:",augmented_inputs.size())
            output = model(aug_inputs)
            m_loss = model.loss_fn(output, aug_labels.float())
            m_loss.backward()
            model.optimizer.step()

            loss += m_loss.item()

        if batch_idx % printBatchEvery == 0 and not printBatchEvery == -1:
            print("\tepoch:", epoch, "batch:", batch_idx, "loss:", m_loss.item(), "running time:", time.time() - s)

    end_time = time.time()
    # print(model)

    if epoch % print_epoch_every == 0:
        if print_accuracy:
            print("epoch", epoch, "average loss:", loss / batch_idx, "accuracy:",
                  test(model, device, test_loader), "% time for epoch:", (end_time - s))
        else:
            print("epoch", epoch, "average loss:", loss / batch_idx, "time for epoch:", (end_time - s))


def test(model, test_loader, device, print_acc=True):
    """
    Run through a test dataset and return the accuracy

    :param model: the network of type torch.nn.Module
    :param test_loader: the training dataset
    :param print_acc: If true accuracy will be printed otherwise it will be returned
    :return: accuracy
    """
    model.eval()

    print('testing received device', device)
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            if Config.interleaving_check:
                print('in test', mp.current_process().name)
                sys.stdout.flush()

            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            if len(list(targets.size())) == 1:
                # each batch item has only one value. this value is the class prediction
                for i in range(list(targets.size())[0]):
                    prediction = round(list(output)[i].item())
                    if prediction == list(targets)[i]:
                        correct += 1

            else:
                # each batch item has num_classes values, the highest of which predicts the class
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)

    if print_acc:
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))

    return acc


def evaluate(model, epochs, device, batch_size=64, augmentor=None):
    """
    Runs all epochs and tests the model after all epochs have run

    :param model: instance of nn.Module
    :param epochs: number of training epochs
    :param batch_size: The dataset batch size
    :return: The trained model
    """
    print('Eval received device', device, 'on processor', mp.current_process())
    train_loader, test_loader = load_data(batch_size)

    s = time.time()
    for epoch in range(1, epochs + 1):
        train(model, train_loader, epoch, test_loader, device, augmentor)
    e = time.time()

    test_acc = test(model, test_loader, device)
    print('Evaluation took', e - s, 'seconds, Test acc:', test_acc)
    return test_acc


def sample_data(device, batch_size=64):
    train_loader, test_loader = load_data(batch_size=batch_size)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        return inputs.to(device), targets.to(device)


def load_data(batch_size=64, dataset=""):
    data_loader_args = {'num_workers': Config.num_workers, 'pin_memory': False if Config.device != 'cpu' else False}
    data_path = DataManager.get_Datasets_folder()
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    download = False

    if dataset == "":
        dataset = Config.dataset.lower()

    #print("loading(",dataset,")data from:",data_path)


    if dataset == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST(data_path,
                           train=True,
                           download=download,
                           transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
            batch_size=batch_size, shuffle=True, **data_loader_args)

        test_loader = DataLoader(
            datasets.MNIST(data_path,
                           train=False,
                           download=download,
                           transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
            batch_size=batch_size, shuffle=True, **data_loader_args)
    elif dataset == 'fassion_mnist':
        train_loader = DataLoader(
            datasets.FashionMNIST(data_path,
                                  train=True, download=download,
                                  transform=image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )

        test_loader = DataLoader(
            datasets.FashionMNIST(data_path,
                                  train=False, download=download,
                                  transform=image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )
    elif dataset == 'cifar10':
        train_loader = DataLoader(
            datasets.CIFAR10(data_path,
                             train=True, download=download,
                             transform=image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )

        test_loader = DataLoader(
            datasets.CIFAR10(data_path,
                             train=False, download=download,
                             transform=image_transform),
            batch_size=batch_size, shuffle=True, **data_loader_args
        )
    else:
        raise Exception('Invalid dataset name, options are fassion_mnist, mnist or cifar')

    return train_loader, test_loader
