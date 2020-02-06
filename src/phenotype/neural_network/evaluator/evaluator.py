from __future__ import annotations

from os.path import join
from typing import TYPE_CHECKING

import random
import time
import wandb

import numpy as np
import torch

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from runs.runs_manager import save_config, get_run_folder_path
from src.phenotype.augmentations.batch_augmentation_scheme import BatchAugmentationScheme
from src.phenotype.neural_network.evaluator.data_loader import imshow, load_data, load_transform
from configuration import config

if TYPE_CHECKING:
    from src.phenotype.neural_network.neural_network import Network

def evaluate(model: Network, n_epochs=config.epochs_in_evolution) -> float:
    """trains model on training data, test on testing and returns test acc"""
    if config.dummy_run and not config.fully_train:
        if config.dummy_time > 0:
            time.sleep(config.dummy_time)
        return random.random()

    aug = None if not config.evolve_da else model.blueprint.get_da().to_phenotype()

    train_loader = load_data(load_transform(aug), 'train')
    test_loader = load_data(load_transform(), 'test') if config.fully_train else None

    device = config.get_device()
    start = config.current_ft_epoch

    for epoch in range(start, n_epochs):
        loss = train_epoch(model, train_loader, aug, device)

        if config.fully_train:
            _fully_train_logging(model, test_loader, loss, epoch)

    test_loader = load_data(load_transform(), 'test') if test_loader is None else test_loader
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
    if config.evolve_da:
        inputs = augmentor(list(inputs.numpy()))

    inputs, labels = inputs.to(device), labels.to(device)
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


def _fully_train_logging(model: Network, test_loader: DataLoader, loss: float, epoch: int):
    print('epoch: {}\nloss: {}'.format(epoch, loss))

    log = {}
    if epoch % config.fully_train_accuracy_test_period == 0:
        acc = test_nn(model, test_loader)
        log['accuracy'] = acc
        print('accuracy: {}'.format(acc))
    print('\n')

    config.current_ft_epoch = epoch
    save_config(config.run_name)

    if config.use_wandb:
        log['loss'] = loss
        wandb.log(log)
        model.save()
        wandb.save(model.save_location())

        wandb.config.current_ft_epoch = epoch
        wandb.save(join(get_run_folder_path(config.run_name), 'config.json'))

