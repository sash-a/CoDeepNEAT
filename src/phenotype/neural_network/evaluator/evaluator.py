from __future__ import annotations

from os.path import join
from typing import TYPE_CHECKING, Union
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import random
import time
import wandb
import torch
import numpy as np
import multiprocessing as mp

from runs.runs_manager import save_config, get_run_folder_path
from src.phenotype.augmentations.batch_augmentation_scheme import BatchAugmentationScheme
from src.phenotype.neural_network.evaluator.data_loader import imshow, load_data, load_transform
from configuration import config, internal_config

if TYPE_CHECKING:
    from src.phenotype.neural_network.neural_network import Network

RETRY = 'retry'


def evaluate(model: Network, n_epochs, training_target=-1, attempt=0) -> Union[float, str]:
    """trains model on training data, test on testing and returns test acc"""
    if config.dummy_run:
        if config.dummy_time > 0:
            time.sleep(config.dummy_time)
        return random.random()

    aug = None if not config.evolve_da else model.blueprint.get_da().to_phenotype()

    train_loader = load_data(load_transform(aug), 'train')
    test_loader = load_data(load_transform(), 'test') if config.fully_train else None

    device = config.get_device()
    start = model.ft_epoch

    if config.fully_train:
        n_epochs = config.fully_train_max_epochs  # max number of epochs for a fully train

    max_acc = 0
    max_acc_age = 0
    for epoch in range(start, n_epochs):
        loss = train_epoch(model, train_loader, aug, device)
        print(f'Process {mp.current_process().name}, epoch {epoch}, blueprint {model.blueprint.id} -> loss: {loss}')

        acc = -1
        test_intermediate_accuracy = config.fully_train and epoch % config.fully_train_accuracy_test_period == 0
        if test_intermediate_accuracy:
            acc = test_nn(model, test_loader)
            if acc > max_acc:
                max_acc = acc
                max_acc_age = 0

            if should_retry_training(max_acc, training_target, epoch):
                # the training is not making target, start again
                # this means that the network is not doing as well as its duplicate in evolution
                return RETRY

            if max_acc_age >= 2:
                # wait 2 accuracy checks, if the max acc has not increased - this network has finished training
                print('training has plateaued stopping')
                return max_acc

            max_acc_age += 1

        if config.fully_train:
            _fully_train_logging(model, loss, epoch, attempt, acc)

    test_loader = load_data(load_transform(), 'test') if test_loader is None else test_loader
    final_test_acc = test_nn(model, test_loader)
    return max(final_test_acc, max_acc)


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


def _fully_train_logging(model: Network, loss: float, epoch: int, attempt: int, acc: float = -1):
    print('epoch: {}\nloss: {}'.format(epoch, loss))

    log = {}
    metric_name = 'accuracy_fm_' + str(model.target_feature_multiplier) + ("_r_" + str(attempt) if attempt > 0 else "")
    if acc != -1:
        log[metric_name] = acc
        print('accuracy: {}'.format(acc))
    print('\n')

    internal_config.ft_epoch = epoch
    save_config(config.run_name)

    if config.use_wandb:
        log['loss_' + str(attempt)] = loss
        wandb.log(log)
        model.save()
        wandb.save(model.save_location())

        wandb.config.update({'current_ft_epoch': epoch}, allow_val_change=True)
        wandb.save(join(get_run_folder_path(config.run_name), 'config.json'))


def should_retry_training(acc, training_target, current_epoch):
    if training_target == -1:
        return False
    progress = current_epoch / config.epochs_in_evolution
    performance = acc / training_target

    """
        by 50% needs 50% ~ with half as many epochs as given in evo - the network should have half the acc it got
        by 100% needs 75%
        by 200% needs 90%
        by 350% needs 100%
    """
    progress_checks = [0.5, 1, 2, 3.5]
    targets = [0.5, 0.75, 0.9, 1]

    print("checking if should retry training. prog:", progress, "perf:", performance)

    for prog_check, target in zip(progress_checks, targets):
        if progress <= prog_check:
            # this is the target to use
            progress_normalised_target = target * progress / prog_check  # linear interpolation of target
            if performance < progress_normalised_target:
                print("net failed to meet target e:", current_epoch, "acc:", acc,
                      "prog:", progress, "prog check:", prog_check, "target:",
                      target, "norm target:", progress_normalised_target)
                return True
            break  # only compare to first fitting target

    return False
