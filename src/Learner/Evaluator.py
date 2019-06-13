# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py

from torch import no_grad
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time


def train(model, device, train_loader, epoch):
    """
    Run a single train epoch

    :param model: the network of type torch.nn.Module
    :param device: Device to train on (cuda or cpu)
    :param train_loader: the training dataset
    :param epoch: the current epoch
    """
    model.train()

    loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        model.optimizer.zero_grad()
        # compute loss without variables to avoid copying from gpu to cpu
        m_loss = model.loss_fn(model(inputs), targets)
        m_loss.backward()
        model.optimizer.step()

        loss += m_loss

    if epoch % 10 == 0:
        print("loss:", loss)


def test(model, device, test_loader):
    """
    Run through a test dataset and return the accuracy

    :param model: the network of type torch.nn.Module
    :param device: Device to train on (cuda or cpu)
    :param test_loader: the training dataset
    :return: accuracy
    """
    model.eval()

    test_loss = 0
    correct = 0
    with no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            test_loss += F.nll_loss(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def evaluate(model, epochs, dataset='mnist', path='../../data', device='cuda', timer=False):
    """
    Runs all epochs and tests the model after all epochs have run

    :param model: instance of nn.Module
    :param epochs: number of training epochs
    :param dataset: Either mnist or imgnet
    :param path: where to store the dataset
    :param device: Either cuda or cpu
    :return: The trained model
    """
    # Make this params
    # num_workers=1 is giving issues, but 0 runs slower
    data_loader_args = {'num_workers': 0, 'pin_memory': True} if device == 'cuda' else {}

    train_loader = None
    test_loader = None
    if dataset.lower() == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST(path,
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=64, shuffle=True, **data_loader_args)

        test_loader = DataLoader(
            datasets.MNIST(path,
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=64, shuffle=True, **data_loader_args)

    elif dataset.lower() == 'imgnet':
        train_loader = DataLoader(
            datasets.ImageNet(path, train=True)  # TODO
        )

        test_loader = DataLoader(
            datasets.ImageNet(path, train=False)  # TODO
        )
    else:
        raise Exception('Invalid dataset name, options are imgnet or mnist')

    s = time.time()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, epoch)
    e = time.time()

    print('Took:', e - s, 'seconds')
    test(model, device, test_loader)
