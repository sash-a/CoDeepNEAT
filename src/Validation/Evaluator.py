# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py
import math
import sys
import time

import torch
import torch.multiprocessing as mp
from data import DataManager

from src.Config import Config
from src.DataAugmentation import BatchAugmentor
from src.Validation.DataLoader import load_data

printBatchEvery = -1  # -1 to switch off batch printing
print_epoch_every = -1  # -1 to switch off epoch printing


def train_epoch(model, train_loader, epoch, test_loader, device, augmentors=None, print_accuracy=False,
                drop_adaptive_learning_rate=False):
    """
    Run a single train epoch

    :param model: the network of type torch.nn.Module
    :param train_loader: the training dataset
    :param epoch: the current epoch
    :param test_loader: The test dataset loader
    :param print_accuracy: True if should test when printing batch info
    """
    model.train()
    loss = 0
    batch_idx = 0
    loops = 1 if not augmentors else 1 + len(augmentors)
    loops = 1 if Config.batch_by_batch else loops

    if Config.drop_learning_rate:
        if not Config.use_adaptive_learning_rate_adjustment:
            learning_rate_coefficient = 1 / pow(Config.drop_factor, math.floor(epoch / Config.drop_period))
            model.multiply_learning_rate(learning_rate_coefficient)
        elif drop_adaptive_learning_rate:
            model.multiply_learning_rate(1 / Config.drop_factor)

    for i in range(loops):
        if i == 0 and not Config.train_on_origonal_data and not Config.batch_by_batch:
            # skip origInal data in epoch splicing
            continue

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # batch loop
            model.optimizer.zero_grad()

            has_augs = augmentors is not None and len(augmentors) > 0

            train_on_aug = has_augs and i >= 1
            train_on_aug = train_on_aug or (Config.batch_by_batch and has_augs)
            if train_on_aug:
                if not Config.batch_by_batch:
                    augmentor = augmentors[i - 1]
                    if augmentor is None:
                        continue
                    loss += train_batch(model, inputs, targets, device, augmentor=augmentor)
                else:
                    for augmentor in augmentors:
                        loss += train_batch(model, inputs, targets, device, augmentor=augmentor)
                        model.optimizer.zero_grad()

            train_on_original = i == 0
            train_on_original = train_on_original or Config.batch_by_batch

            if train_on_original:
                loss += train_batch(model, inputs, targets, device)

    if print_epoch_every != -1 and epoch % print_epoch_every == 0:
        if print_accuracy:
            test_acc = test(model, test_loader, device, print_acc=False)
            print("epoch", epoch, "average loss:", loss / batch_idx, "accuracy:", test_acc, "i = ", i)
            with open(DataManager.get_results_file(), 'a+') as f:
                f.write(repr(epoch) + ': ' + repr(test_acc) + ', loss: ' + repr(loss / batch_idx))
                f.write('\n')

            model.train()
            return test_acc

        else:
            print("epoch", epoch, "average loss:", loss / batch_idx, "i=", i)

    end_time = time.time()


def train_batch(model, inputs, targets, device, augmentor=None):
    """trains the model for one batch """
    if augmentor is not None:
        train_inputs, train_labels = BatchAugmentor.augment_batch(inputs.numpy(), targets.numpy(), augmentor)
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    else:
        train_inputs, train_labels = inputs.to(device), targets.to(device)

    output = model(train_inputs)
    m_loss = model.loss_fn(output, train_labels)
    m_loss.backward()
    model.optimizer.step_evolution()

    return m_loss.item()


def test(model, test_loader, device, print_acc=False):
    """
    Run through a test dataset and return the accuracy

    :param model: the network of type torch.nn.Module
    :param test_loader: the training dataset
    :param print_acc: If true accuracy will be printed otherwise it will be returned
    :return: accuracy
    """
    model.eval()

    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            if len(list(output.size())) == 1:
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
        print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), acc))

    return acc


def evaluate(model, epochs, device, batch_size=Config.batch_size, augmentors=None, train_loader=None, test_loader=None,
             print_accuracy=False, training_target=-1):
    """
    Runs all epochs and tests the model after all epochs have run

    :param model: instance of nn.Module
    :param epochs: number of training epochs
    :param batch_size: The dataset batch size
    :return: The trained model
    """
    if train_loader is None:
        train_loader, test_loader = load_data(batch_size)

    s = time.time()
    max_acc = 0
    time_with_max_acc = 0
    for epoch in range(1, epochs + 1):
        adapt_learning_rate = Config.use_adaptive_learning_rate_adjustment and time_with_max_acc >= 2

        response = train_epoch(model, train_loader, epoch, test_loader, device, augmentors,
                               print_accuracy=print_accuracy, drop_adaptive_learning_rate=adapt_learning_rate)
        if adapt_learning_rate:
            time_with_max_acc = 3

        if response and response > max_acc:
            time_with_max_acc = 0
            max_acc = response
        else:
            time_with_max_acc += 1

        if Config.toss_bad_runs and training_target != -1:
            """by epoch 5, run must be at 50% of target
                by epoch 10, run must be at 75% of target
                by epoch 25 run must be at 90% of target
                by epoch 50 run must be at target
            """
            targets = {5: 0.5, 10: 0.75, 25: 0.9, 50: 1}
            target = targets[epoch] if epoch in targets else 0
            target *= training_target
            if response is not None and max_acc < target:
                print("target", target, "missed(", max_acc, "), tossing train")
                return "toss"

    e = time.time()

    test_acc = test(model, test_loader, device)
    return test_acc
