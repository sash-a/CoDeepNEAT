from __future__ import annotations

import os
from typing import TYPE_CHECKING
import wandb

import src.main.singleton as S

from src.analysis.run import get_run
from configuration import config, internal_config
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src.phenotype.neural_network.evaluator.evaluator import evaluate, RETRY
from src.phenotype.neural_network.neural_network import Network

if TYPE_CHECKING:
    from src.analysis.run import Run

MAX_RETRIES = 5


def fully_train(run_name, epochs, n=1):
    """
    Loads and trains from a saved run
    :param run_name: name of the old run
    :param epochs: number of epochs to train the best networks for
    :param n: number of the best networks to train
    """
    print('Fully training...')
    run: Run = get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    in_size = get_data_shape()

    for blueprint, gen_num in best_blueprints:
        for feature_multiplier in config.ft_feature_multipliers:
            accuracy = RETRY
            remaining_retries = MAX_RETRIES
            while accuracy == RETRY and remaining_retries >= 0:
                model: Network = _create_model(run, blueprint, gen_num, in_size, epochs, feature_multiplier)

                if config.resume_fully_train and os.path.exists(model.save_location()):
                    model = _load_model(blueprint, run, gen_num, in_size)

                if config.use_wandb:
                    wandb.watch(model, criterion=model.loss_fn, log='all', idx=blueprint.id)

                if remaining_retries > 0:
                    accuracy = evaluate(model, epochs, blueprint.max_acc, MAX_RETRIES - remaining_retries)
                else:  # give up retrying, take whatever is produced from training
                    accuracy = evaluate(model, epochs)

                if accuracy == RETRY:
                    print("retrying fully training")
                    internal_config.ft_epoch = 0

                remaining_retries -= 1

            print('Achieved a final accuracy of: {}'.format(accuracy * 100))

    internal_config.finished = True
    internal_config.state = 'finished'


def _create_model(run: Run, blueprint: BlueprintGenome, gen_num, in_size,
                  epochs, feature_multiplier) -> Network:
    S.instance = run.generations[gen_num]
    modules = run.get_modules_for_blueprint(blueprint)
    model: Network = Network(blueprint, in_size, sample_map=blueprint.best_module_sample_map,
                             feature_multiplier=feature_multiplier, allow_module_map_ignores=False).to(
        config.get_device())

    print("Blueprint: {}\nModules: {}\nSample map: {}\n Species used: {}"
          .format(blueprint,
                  modules,
                  blueprint.best_module_sample_map,
                  list(set(
                      [blueprint.nodes[node_id].species_id for node_id in blueprint.get_fully_connected_node_ids()]))))
    print("Training model which scored: {} in evolution for {} epochs, with {} parameters\n"
          .format(blueprint.max_acc, epochs, model.size()))

    return model


def _load_model(dummy_bp: BlueprintGenome, run: Run, gen_num: int, in_size) -> Network:
    if not config.resume_fully_train:
        raise Exception('Calling resume training, but config.resume_fully_train is false')

    S.instance = run.generations[gen_num]
    model: Network = Network(dummy_bp, in_size, sample_map=dummy_bp.best_module_sample_map,
                             allow_module_map_ignores=False).to(config.get_device())
    model.load()

    return model
