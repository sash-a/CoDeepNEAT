from __future__ import annotations

from typing import TYPE_CHECKING, Union

import random
import time

import numpy as np
import torch

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.analysis.reporters.base_reporter import BaseReporter
from src.analysis.reporters.reporter_set import ReporterSet
from src.phenotype.augmentations.batch_augmentation_scheme import BatchAugmentationScheme
from src.phenotype.neural_network.evaluator.data_loader import imshow, load_data, load_transform
from configuration import config

from src.phenotype.neural_network.evaluator.eval_utils import fetch_training_instruction, \
    RETRY, CONTINUE, STOP, DROP_LR
from src.phenotype.neural_network.evaluator.training_results import TrainingResults
from src.utils.wandb_utils import _fully_train_logging

if TYPE_CHECKING:
    from src.phenotype.neural_network.neural_network import Network


def evaluate(model: Network, n_epochs, training_target=-1, attempt=0, reporter=ReporterSet()) -> Union[float, str]:
    """trains model on training data, test on testing and returns test acc"""
    if config.dummy_run:
        if config.dummy_time > 0:
            time.sleep(config.dummy_time)
        return random.random()

    aug = None if not config.evolve_da else model.blueprint.get_da().to_phenotype()

    train_loader = load_data(load_transform(aug), 'train')
    test_loader = load_data(load_transform(), 'test')

    device = config.get_device()
    start = model.last_epoch

    training_results = TrainingResults()
    for epoch in range(start, n_epochs):
        reporter.on_start_epoch(model, epoch)

        loss = train_epoch(model, train_loader, aug, device, reporter)
        model.last_epoch = epoch
        training_results.add_loss(loss)

        acc = -1
        needs_intermediate_acc = config.ft_auto_stop_count != -1 or config.ft_retries or config.ft_allow_lr_drops
        test_intermediate_accuracy = config.fully_train and epoch % config.fully_train_accuracy_test_period == 0 \
                                     and needs_intermediate_acc

        if test_intermediate_accuracy:
            acc = test_nn(model, test_loader)
            training_results.add_accuracy(acc, epoch)

            # if config.fully_train:
            #     _fully_train_logging(model, loss, epoch, attempt, acc)
        reporter.on_end_epoch(model, epoch, loss, acc)

        TRAINING_INSTRUCTION = fetch_training_instruction(training_results, training_target)
        if TRAINING_INSTRUCTION == CONTINUE:
            continue
        if TRAINING_INSTRUCTION == RETRY:
            return RETRY
        if TRAINING_INSTRUCTION == STOP:
            if len(training_results.accuracies) > 0:
                return training_results.get_max_acc()
            else:
                break  # exits for, runs final acc test, returns
        if TRAINING_INSTRUCTION == DROP_LR:
            model.drop_lr()

    print("doing final acc test")
    final_test_acc = test_nn(model, test_loader)
    return max(final_test_acc, training_results.get_max_acc())


def train_epoch(model: Network, train_loader: DataLoader, augmentor: BatchAugmentationScheme, device, reporter: ReporterSet) -> float:
    model.train()
    loss: float = 0
    batch_idx = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if config.max_batches != -1 and batch_idx > config.max_batches:
            break

        model.optimizer.zero_grad()

        reporter.on_start_batch(batch_idx, loss)
        loss += train_batch(model, inputs, targets, augmentor, device)
        reporter.on_end_batch(batch_idx, loss)

    return loss / batch_idx


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

    return m_loss.item() / config.batch_size


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
