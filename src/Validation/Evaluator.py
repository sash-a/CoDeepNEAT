# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py
import sys

import torch
from src.DataAugmentation import BatchAugmentor
from src.Config import Config

import time
import torch.multiprocessing as mp

from src.Validation.DataLoader import load_data

printBatchEvery = -1  # -1 to switch off batch printing
print_epoch_every = -1  # -1 to switch off epoch printing


def train(model, train_loader, epoch, test_loader, device, augmentors=None, print_accuracy=False):
    """
    Run a single train epoch

    :param model: the network of type torch.nn.Module
    :param train_loader: the training dataset
    :param epoch: the current epoch
    :param test_loader: The test dataset loader
    :param print_accuracy: True if should test when printing batch info
    """
    # print('Train received device:', device)
    model.train()

    loss = 0
    batch_idx = 0
    loops = 1  if not Config.evolve_data_augmentations else 1 + len(augmentors)

    s = time.time()
    for i in range(loops):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            model.optimizer.zero_grad()

            if Config.interleaving_check:
                print('in train', mp.current_process().name)
                sys.stdout.flush()

            if augmentors is not None and len(augmentors) > 0 and i >= 1:
                augmentor = augmentors[i-1]
                if augmentor is None:
                    continue
                # print("augmenting batch with:", augmentor)
                aug_inputs, aug_labels = BatchAugmentor.augment_batch(inputs.numpy(), targets.numpy(), augmentor)
                # print("augmented batch")
                # print("targets:",targets,"\naug_targets:",aug_labels)
                # print("inputs:",inputs,"aug_inputs:", aug_inputs)
                # print("shape origonal:",inputs.size(), "aug:",aug_inputs.size())
                # print("range of augs~",torch.min(aug_inputs),":",torch.max(aug_inputs))
                # print("Vectors", inputs-aug_inputs)

                aug_inputs, aug_labels = aug_inputs.to(device), aug_labels.to(device)
                # print("training on augmented images shape:",augmented_inputs.size())
                output = model(aug_inputs)
                m_loss = model.loss_fn(output, aug_labels)
                m_loss.backward()
                model.optimizer.step()

            if i == 0:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                m_loss = model.loss_fn(output, targets)
                m_loss.backward()
                model.optimizer.step()
                loss += m_loss.item()

            if batch_idx % printBatchEvery == 0 and not printBatchEvery == -1:
                print("\tepoch:", epoch, "batch:", batch_idx, "loss:", m_loss.item(), "running time:", time.time() - s,"i=",i)

        if print_epoch_every != -1 and epoch % print_epoch_every == 0:
            if print_accuracy:
                print("epoch", epoch, "average loss:", loss / batch_idx, "accuracy:",
                      test(model, test_loader, device, print_acc=False), "i = ", i )
                model.train()

            else:
                print("epoch", epoch, "average loss:", loss / batch_idx, "i=",i)

    end_time = time.time()
    # print(model)




def test(model, test_loader, device, print_acc=False):
    """
    Run through a test dataset and return the accuracy

    :param model: the network of type torch.nn.Module
    :param test_loader: the training dataset
    :param print_acc: If true accuracy will be printed otherwise it will be returned
    :return: accuracy
    """
    model.eval()

    # print('testing received device', device)
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            if Config.interleaving_check:
                print('in test', mp.current_process().name)
                sys.stdout.flush()

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


def evaluate(model, epochs, device, batch_size=64, augmentors=None, train_loader=None, test_loader=None,
             print_accuracy=False):
    """
    Runs all epochs and tests the model after all epochs have run

    :param model: instance of nn.Module
    :param epochs: number of training epochs
    :param batch_size: The dataset batch size
    :return: The trained model
    """
    # print('Eval received device', device, 'on processor', mp.current_process())
    # print("got das:",augmentors)
    if train_loader is None:
        train_loader, test_loader = load_data(batch_size)

    s = time.time()
    for epoch in range(1, epochs + 1):
        train(model, train_loader, epoch, test_loader, device, augmentors, print_accuracy=print_accuracy)
    e = time.time()

    test_acc = test(model, test_loader, device)
    # print('Evaluation took', e - s, 'seconds, Test acc:', test_acc, end='\n')
    return test_acc
