from __future__ import annotations

from typing import TYPE_CHECKING
import wandb

import src2.main.singleton as S

from src2.analysis.run import get_run
from src2.configuration import config
from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src2.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src2.phenotype.neural_network.evaluator.evaluator import evaluate
from src2.phenotype.neural_network.neural_network import Network

if TYPE_CHECKING:
    from src2.analysis.run import Run


def fully_train(run_name, n=1, epochs=100):
    """
    Loads and trains from a saved run
    :param run_name: name of the old run
    :param n: number of the best networks to train
    :param epochs: number of epochs to train the best networks for
    """
    run: Run = get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    in_size = get_data_shape()

    if config.resume_fully_train:
        model = _load_model(best_blueprints[0][0], run, in_size)
        accuracy = evaluate(model, num_epochs=epochs, fully_training=True)
        print('Achieved a final accuracy of: {}'.format(accuracy * 100))
    else:
        for blueprint, gen_number in best_blueprints:
            model = _create_model(run, blueprint, gen_number, in_size, epochs)
            accuracy = evaluate(model, num_epochs=epochs, fully_training=True)
            print('Achieved a final accuracy of: {}'.format(accuracy * 100))


def _create_model(run: Run, blueprint: BlueprintGenome, gen_number, in_size, epochs) -> Network:
    S.instance = run.generations[gen_number]
    modules = run.get_modules_for_blueprint(blueprint)
    model: Network = Network(blueprint, in_size, sample_map=blueprint.best_module_sample_map).to(config.device)

    if config.use_wandb:
        wandb.watch(model, criterion=model.loss_fn, log='all', idx=blueprint.id)

    print("Blueprint: {}\nModules: {}\nSample map: {}\n Species used: {}"
          .format(blueprint,
                  modules,
                  blueprint.best_module_sample_map,
                  list(set([x.species_id for x in blueprint.nodes.values()]))))
    print("Training model which scored: {} in evolution for {} epochs, with {}"
          .format(blueprint.accuracy, epochs, model.size()))

    return model


def _load_model(dummy_bp: BlueprintGenome, run: Run, in_size):
    if not config.resume_fully_train:
        raise Exception('Calling resume training, but config.resume_fully_train is false')

    S.instance = run.generations[0]
    model: Network = Network(dummy_bp, in_size, sample_map=dummy_bp.best_module_sample_map).to(config.device)
    model.load()
