from __future__ import annotations

from typing import TYPE_CHECKING

import random
import sys
import time
import wandb

import numpy as np
import torch
import torch.multiprocessing as mp

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src2.phenotype.augmentations.batch_augmentation_scheme import BatchAugmentationScheme
from src2.phenotype.neural_network.evaluator.data_loader import imshow, load_data, load_transform
from src2.configuration import config

if TYPE_CHECKING:
    from src2.phenotype.neural_network.neural_network import Network


def evaluate(model: Network, num_epochs=config.epochs_in_evolution, fully_training=False) -> float:
    """trains model on training data, test on testing and returns test acc"""
    if config.dummy_run and not fully_training:
        if config.dummy_time > 0:
            time.sleep(config.dummy_time)
        return random.random()

    train_loader = load_data(load_transform(model if not config.batch_augmentation else None), 'train')
    device = config.get_device()

    for epoch in range(num_epochs):
        if config.threading_test:
            print('Thread %s bp: %i epoch: %i' % (mp.current_process().name, model.blueprint.id, epoch))
        loss = train_epoch(model, train_loader, model.blueprint.get_da().to_phenotype(), device)

        if fully_training:
            # Save and log if fully training
            print('epoch: {} got loss: {}'.format(epoch, loss))
            if config.use_wandb:
                wandb.log({'loss': loss})
                model.save()
                wandb.save(model.save_location())

    test_loader = load_data(load_transform(), 'test' if not config.fully_train else 'validation')
    return test_nn(model, test_loader)


def train_epoch(model: Network, train_loader: DataLoader, augmentor: BatchAugmentationScheme, device) -> float:
    model.train()
    loss: float = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if config.max_batches != -1 and batch_idx > config.max_batches:
            break
        model.optimizer.zero_grad()
        loss += train_batch(model, inputs, targets, augmentor, device)

    return loss


def train_batch(model: Network, inputs: torch.Tensor, labels: torch.Tensor, augmentor: BatchAugmentationScheme, device):
    if config.threading_test:
        print('training batch on thread:', mp.current_process().name)
        sys.stdout.flush()

    inputs = augmentor(list(inputs.numpy()))
    inputs, labels = inputs.to(device), labels.to(device)
    imshow(inputs[0])
    if config.view_batch_image:
        imshow(inputs[0])

    output = model(inputs)
    m_loss = model.loss_fn(output, labels)
    m_loss.backward()
    model.optimizer.step()

    return m_loss.item()


def test_nn(model: Network, test_loader: DataLoader):
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
