from __future__ import annotations

from typing import TYPE_CHECKING

from configuration import config

if TYPE_CHECKING:
    from src.phenotype.neural_network.evaluator.training_results import TrainingResults

CONTINUE = "continue"
RETRY = "retry"
STOP = "stop"
DROP_LR = "drop_lr"


def handle_accuracy_reading(training_results: TrainingResults, training_target):
    """
        if in evolution, and the config specifies to continue training until the
        loss improvement gradient drops below a threshold -
        a hyperbolic function is fitted over the loss data, and used to estimate the current loss gradient

        if in fully train mode the training continues until the accuracy appears to have plateaued.
        if config specifies, then an acc plateau can also trigger a drop in the LR
    """

    if config.fully_train:
        if check_should_retry_training(training_results.get_max_acc(), training_target, training_results.accuracy_epochs[-1]):
            return RETRY

        max_acc_age = training_results.get_max_acc_age()
        if config.ft_allow_lr_drops:
            if max_acc_age >= 3:
                return STOP
            if max_acc_age == 1: # the first time the max acc stagnates - drop the LR
                return DROP_LR
        else:
            if max_acc_age >=2:
                return STOP

    else:
        """in evolution"""
        if config.loss_based_stopping_in_evolution and len(training_results.losses) > 2:
            current_loss_gradient = training_results.get_loss_gradient()
            # the slope will start highly negative, and move towards zero
            if -current_loss_gradient < config.loss_gradient_threshold:
                return STOP

    return CONTINUE


def check_should_retry_training(acc, training_target, current_epoch):
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

    print("checking if should retry training. prog:",progress,"perf:",performance)

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


