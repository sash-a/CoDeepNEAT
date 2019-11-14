from __future__ import annotations

import random

import torch
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Evaluator.DataLoader import load_data

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network


def evaluate(model: Network, num_epochs=config.epochs_in_evolution):
    """trains model on training data, test on testing and returns test acc"""

    # TODO add in augmentations
    composed_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for epoch in range(num_epochs):
        train_epoch(model, load_data(composed_transform, 'train'))

    return get_test_acc(model, load_data(composed_transform, 'test'))


def train_epoch(model: Network, train_loader: DataLoader):
    model.train()
    loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.optimizer.zero_grad()
        loss += train_batch(model, inputs, targets)


def train_batch(model: Network, inputs: torch.tensor, labels: torch.tensor):
    device = config.get_device()
    inputs, labels = inputs.to(device), labels.to(device)

    output = model(inputs)
    m_loss = model.loss_fn(output, labels)
    m_loss.backward()
    model.optimizer.step()

    return m_loss.item()


def get_test_acc(model: Network, test_loader: DataLoader):
    model.eval()

    with torch.no_grad():
        pass
        # todo find proper way to do this. manual counting logic is slow

    return random.random()
