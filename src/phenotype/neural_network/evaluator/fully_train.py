from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union
import wandb

import src.main.singleton as singleton

from src.analysis.run import get_run
from configuration import config, internal_config
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src.phenotype.neural_network.evaluator.evaluator import evaluate, RETRY
from src.phenotype.neural_network.feature_multiplication import get_model_of_target_size
from src.phenotype.neural_network.neural_network import Network
from src.utils.mp_utils import get_evaluator_pool

if TYPE_CHECKING:
    from src.analysis.run import Run

MAX_RETRIES = 5


def fully_train(run_name, n=1):
    """
    Loads and trains from a saved run
    :param run_name: name of the old run
    :param n: number of the best networks to train
    """
    print('Fully training...')
    run: Run = get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    in_size = get_data_shape()

    for blueprint, gen_num in best_blueprints:
        accuracies = []

        with get_evaluator_pool(singleton.instance) as pool:
            for target_fm in config.ft_feature_multipliers:
                accuracies.append(pool.submit(evaluate_with_retries, run, blueprint, gen_num, in_size, target_fm))

        for accuracy in accuracies:
            print(accuracy)  # Need to consume this list for the process to run

    internal_config.finished = True
    internal_config.state = 'finished'


def _create_model(run: Run, blueprint: BlueprintGenome, gen_num, in_size,
                  target_feature_multiplier) -> Network:
    singleton.instance = run.generations[gen_num]
    modules = run.get_modules_for_blueprint(blueprint)
    model: Network = Network(blueprint, in_size, sample_map=blueprint.best_module_sample_map,
                             allow_module_map_ignores=False, feature_multiplier=1,
                             target_feature_multiplier=target_feature_multiplier).to(config.get_device())
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sample_map = model.sample_map

    if target_feature_multiplier != 1:
        model = get_model_of_target_size(blueprint, sample_map, model_size, in_size,
                                         target_size=model_size * target_feature_multiplier)
        model.target_feature_multiplier = target_feature_multiplier

    print("Blueprint: {}\nModules: {}\nSample map: {}\n Species used: {}"
          .format(blueprint,
                  modules,
                  blueprint.best_module_sample_map,
                  list(set(
                      [blueprint.nodes[node_id].species_id for node_id in blueprint.get_fully_connected_node_ids()]))))
    print("Training model which scored: {} in evolution , with {} parameters with feature mult: {}\n"
          .format(blueprint.max_acc, model.size(), target_feature_multiplier))

    return model


def _load_model(dummy_bp: BlueprintGenome, run: Run, gen_num: int, in_size) -> Network:
    if not config.resume_fully_train:
        raise Exception('Calling resume training, but config.resume_fully_train is false')

    singleton.instance = run.generations[gen_num]
    model: Network = Network(dummy_bp, in_size, sample_map=dummy_bp.best_module_sample_map,
                             allow_module_map_ignores=False).to(config.get_device())
    model.load()

    return model


def evaluate_with_retries(run: Run, blueprint: BlueprintGenome, gen_num: int, in_size,
                          target_feature_multiplier: float):
    accuracy = RETRY
    remaining_retries = MAX_RETRIES
    while accuracy == RETRY and remaining_retries >= 0:
        accuracy = attempt_evaluate(run,
                                    blueprint,
                                    gen_num,
                                    in_size,
                                    target_feature_multiplier,
                                    MAX_RETRIES - remaining_retries)

        if accuracy == RETRY:
            print("retrying fully training")
            internal_config.ft_epoch = 0

        remaining_retries -= 1

    if accuracy == RETRY:
        # Final retry
        accuracy = evaluate(_create_model(run, blueprint, gen_num, in_size, target_feature_multiplier))

    print('Achieved a final accuracy of: {}'.format(accuracy * 100))
    return accuracy


def attempt_evaluate(run: Run, blueprint: BlueprintGenome, gen_num: int, in_size,
                     target_feature_multiplier: float, attempt_number) -> Union[float, str]:
    model: Network = _create_model(run, blueprint, gen_num, in_size, target_feature_multiplier)

    if config.resume_fully_train and os.path.exists(model.save_location()):
        model = _load_model(blueprint, run, gen_num, in_size)

    if config.use_wandb:
        wandb.watch(model, criterion=model.loss_fn, log='all', idx=blueprint.id)

    accuracy = evaluate(model, training_target=blueprint.max_acc, attempt=attempt_number)

    return accuracy
