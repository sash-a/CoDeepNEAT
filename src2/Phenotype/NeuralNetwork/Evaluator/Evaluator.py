from __future__ import annotations

import random
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Evaluator.DataLoader import load_data

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network


def evaluate(model: Network, num_epochs=config.epochs_in_evolution, fully_training=False):
    """trains model on training data, test on testing and returns test acc"""
    if config.dummy_run and not fully_training:
        if config.dummy_time > 0:
            time.sleep(config.dummy_time)
        return random.random()

    # TODO add in augmentations
    if config.evolve_data_augmentations:
        composed_transform = transforms.Compose([
            # TODO dunno how we gonna implement da scheme, but this avoids passing any DA schemes around cause it lives in the bp in the model
            model.blueprint.da_scheme.to_phenotype(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # TODO why is this commented out
        ])
    else:
        composed_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_loader = load_data(composed_transform, 'train')
    device = config.get_device()
    for epoch in range(num_epochs):
        if config.threading_test:
            print('Thread %s bp: %i epoch: %i' % (threading.current_thread().name[-1], model.blueprint.id, epoch))
        train_epoch(model, train_loader, device)

    test_loader = load_data(composed_transform, 'test')
    return get_test_acc(model, test_loader)


def train_epoch(model: Network, train_loader: DataLoader, device, max_batches=-1):
    model.train()
    loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if max_batches != -1 and batch_idx > max_batches:
            break
        model.optimizer.zero_grad()
        loss += train_batch(model, inputs, targets, device)


def train_batch(model: Network, inputs: torch.tensor, labels: torch.tensor, device):
    if config.threading_test:
        print('training batch on thread:', threading.current_thread().name)
        sys.stdout.flush()

    inputs, labels = inputs.to(device), labels.to(device)

    output = model(inputs)
    m_loss = model.loss_fn(output, labels)
    m_loss.backward()
    model.optimizer.step()

    return m_loss.item()


def get_test_acc(model: Network, test_loader: DataLoader):
    model.eval()

    count = 0
    total_acc = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(config.get_device()), targets.to(config.get_device())

            output = model(inputs)

            softmax = torch.exp(output).cpu()
            prob = list(softmax.numpy())
            predictions = np.argmax(prob, axis=1)

            acc = accuracy_score(targets.cpu(), predictions)
            total_acc += acc
            count = batch_idx

    return total_acc / count
