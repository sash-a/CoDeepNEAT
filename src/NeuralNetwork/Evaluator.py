# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.DataAugmentation import BatchAugmentor
from src.Config import Config

import time

printBatchEvery = -1  # -1 to switch off batch printing
print_epoch_every = 1


def train(model, train_loader, epoch, test_loader, augmentor=None, print_accuracy=False):
    """
    Run a single train epoch

    :param model: the network of type torch.nn.Module
    :param train_loader: the training dataset
    :param epoch: the current epoch
    :param test_loader: The test data set loader
    :param print_accuracy: True if should test when printing batch info
    """
    model.train()
    device = Config.device

    loss = 0
    batch_idx = 0

    s = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if augmentor is not None:
            aug_inputs, aug_labels = BatchAugmentor.augment_batch(inputs.numpy(), targets.numpy(), augmentor)
        inputs, targets = inputs.to(device), targets.to(device)

        model.optimizer.zero_grad()

        output = model(inputs)
        m_loss = model.loss_fn(output, targets.float())
        # del inputs
        # del targets
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


def test(model, test_loader, print_acc=True):
    """
    Run through a test dataset and return the accuracy

    :param model: the network of type torch.nn.Module
    :param device: Device to train on (cuda or cpu)
    :param test_loader: the training dataset
    :param print_acc: If true accuracy will be printed otherwise it will be returned
    :return: accuracy
    """
    model.eval()
    device = Config.device

    test_loss = 0
    correct = 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            if len(list(targets.size())) == 1:
                # each batch item has only one value. this value is the class prediction
                test_loss += loss_fn(output, targets.float())
                for i in range(list(targets.size())[0]):
                    prediction = round(list(output)[i].item())
                    if prediction == list(targets)[i]:
                        correct += 1

            else:
                # each batch item has num_classes values, the highest of which predicts the class
                test_loss += F.nll_loss(output, targets, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    if print_acc:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), acc))

    return acc


def evaluate(model, epochs, dataset='mnist', path='../../data', batch_size=64, augmentor=None):
    """
    Runs all epochs and tests the model after all epochs have run

    :param model: instance of nn.Module
    :param epochs: number of training epochs
    :param dataset: Either mnist or imgnet
    :param path: where to store the dataset
    :param device: Either cuda or cpu
    :param batch_size: The data set batch size
    :return: The trained model
    """
    # Make this params
    # num_workers=1 is giving issues, but 0 runs slower
    data_loader_args = {'num_workers': 0, 'pin_memory': True} if Config.device == 'cuda' else {}

    train_loader, test_loader = load_data(dataset, path, batch_size)

    s = time.time()
    for epoch in range(1, epochs + 1):
        train(model, train_loader, epoch, test_loader, augmentor)
    e = time.time()

    test_acc = test(model, test_loader)
    print('Evaluation took', e - s, 'seconds, Test acc:', test_acc)
    return test_acc


def sample_data(dataset='mnist', path='../../data', batch_size=64):
    train_loader, test_loader = load_data(dataset, path, batch_size)
    device = Config.device
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        return inputs.to(device), targets.to(device)


def load_data(dataset, path, batch_size=64):
    # data_loader_args = {'num_workers': 0, 'pin_memory': True} if device == 'cuda' else {}
    data_loader_args = {}

    if dataset.lower() == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST(path,
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **data_loader_args)

        test_loader = DataLoader(
            datasets.MNIST(path,
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **data_loader_args)

    elif dataset.lower() == 'imgnet':
        train_loader = DataLoader(
            datasets.ImageNet(path, train=True)  # TODO
        )

        test_loader = DataLoader(
            datasets.ImageNet(path, train=False)  # TODO
        )
    else:
        raise Exception('Invalid dataset name, options are imgnet or mnist')

    return train_loader, test_loader
