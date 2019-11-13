from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from src2.Configuration import config

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network


def evaluate(model: Network, num_epochs=config.epochs_in_evolution, batch_size=config.batch_size):
    """trains model on training data, test on testing and returns test acc"""

    for epoch in range(num_epochs):
        train_epoch(model)

    return get_test_acc(model)


def train_epoch(model: Network, train_loader: DataLoader):
    model.train()
    loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.optimizer.zero_grad()
        loss += train_batch(model, inputs, targets)


def train_batch(model: Network, input: torch.tensor, labels: torch.tensor):
    output = model(input)
    m_loss = model.loss_fn(output, labels)
    m_loss.backward()
    model.optimizer.step()

    return m_loss.item()


def get_test_acc(model: Network, test_loader: DataLoader):
    model.eval()

    with torch.no_grad():
        pass
        # todo find proper way to do this. manual counting logic is slow
