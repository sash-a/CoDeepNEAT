# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py

from torch import nn, no_grad
import torch.nn.functional as F


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
        # TODO add all to GPU in parallel and keep the inputs there
        inputs, targets = inputs.to(device), targets.to(device)
        model.optimizer.zero_grad()
        # compute loss without variables to avoid copying from gpu to cpu
        m_loss = model.loss_fn(model(inputs), targets)
        m_loss.backward()
        model.optimizer.step()
        loss += m_loss


    if epoch % 1 == 0:
        print("loss:",loss)


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
