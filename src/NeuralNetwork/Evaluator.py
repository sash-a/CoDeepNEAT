# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
from torch import no_grad, nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.DataAugmentation import placeholder

import time

printBatchEvery = 150  # -1 to switch off batch printing
print_epoch_every = 1


def train(model, device, train_loader, epoch, test_loader, print_accuracy=True):
    """
    Run a single train epoch

    :param model: the network of type torch.nn.Module
    :param device: Device to train on (cuda or cpu)
    :param train_loader: the training dataset
    :param epoch: the current epoch
    :param test_loader: The test data set loader
    :param print_accuracy: True if should test when printing batch info
    """
    model.train()

    s = time.time()
    loss = 0
    batch_idx = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        model.optimizer.zero_grad()
        augmented_inputs, augmented_labels = placeholder.augment_batch(inputs,targets)

        output = model(inputs)
        m_loss = model.loss_fn(output, targets.float())
        m_loss.backward()
        model.optimizer.step()

        loss += m_loss.item()

        if(not augmented_inputs is None):
            output = model(augmented_inputs)
            m_loss = model.loss_fn(output, augmented_labels.float())
            m_loss.backward()
            model.optimizer.step()

        if batch_idx % printBatchEvery == 0 and not printBatchEvery == -1:
            print("\tepoch:", epoch, "batch:", batch_idx, "loss:", m_loss.item(), "running time:", time.time() - s)

    end_time = time.time()
    if epoch % print_epoch_every == 0:
        if print_accuracy:
            print("epoch", epoch, "average loss:", loss / batch_idx, "accuracy:",
                  test(model, device, test_loader, print_acc=False), "% time for epoch:", (end_time - s))
        else:
            print("epoch", epoch, "average loss:", loss / batch_idx, "time for epoch:", (end_time - s))


def test(model, device, test_loader, print_acc=True):
    """
    Run through a test dataset and return the accuracy

    :param model: the network of type torch.nn.Module
    :param device: Device to train on (cuda or cpu)
    :param test_loader: the training dataset
    :param print_acc: If true accuracy will be printed otherwise it will be returned
    :return: accuracy
    """
    model.eval()

    test_loss = 0
    correct = 0
    loss_fn = nn.MSELoss()
    with no_grad():
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

    if print_acc:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    else:
        return 100. * correct / len(test_loader.dataset)


def evaluate(model, epochs, dataset='mnist', path='../../data', device=torch.device('cuda:0'), timer=False,
             batch_size=64):
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
    data_loader_args = {'num_workers': 0, 'pin_memory': True} if device == 'cuda' else {}

    train_loader, test_loader = load_data('mnist','../../data')

    s = time.time()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, epoch, test_loader)
    e = time.time()

    print('Took:', e - s, 'seconds')
    return test(model, device, test_loader)


def sample_data(dataset='mnist', path='../../data', device=torch.device('cuda:0'), batchSize = 64):
    train_loader, test_loader = load_data('mnist','../../data')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        return inputs.to(device), targets.to(device)


def load_data(dataset, path, batchSize = 64):
    #data_loader_args = {'num_workers': 0, 'pin_memory': True} if device == 'cuda' else {}
    data_loader_args = {}

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
            batch_size=batchSize, shuffle=True, **data_loader_args)

        test_loader = DataLoader(
            datasets.MNIST(path,
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batchSize, shuffle=True, **data_loader_args)

    elif dataset.lower() == 'imgnet':
        train_loader = DataLoader(
            datasets.ImageNet(path, train=True)  # TODO
        )

        test_loader = DataLoader(
            datasets.ImageNet(path, train=False)  # TODO
        )
    else:
        raise Exception('Invalid dataset name, options are imgnet or mnist')

    return train_loader,test_loader
