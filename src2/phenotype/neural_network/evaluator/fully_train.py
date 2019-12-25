from __future__ import annotations

from typing import TYPE_CHECKING
import wandb

import src2.main.singleton as S

from src2.analysis.run import get_run
from src2.configuration import config
from src2.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src2.phenotype.neural_network.evaluator.evaluator import evaluate
from src2.phenotype.neural_network.neural_network import Network

if TYPE_CHECKING:
    from src2.analysis.run import Run


def fully_train(run_name, n=1, num_epochs=100):
    """
    Loads and trains from a saved run
    :param run_name: name of the old run
    :param n: number of the best networks to train
    :param num_epochs: number of epochs to train the best networks for
    """
    run: Run = get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    in_size = get_data_shape()
    device = config.get_device()

    for blueprint, gen_number in best_blueprints:
        S.instance = run.generations[gen_number]
        modules = run.get_modules_for_blueprint(blueprint)
        model: Network = Network(blueprint, in_size, sample_map=blueprint.best_module_sample_map).to(device)

        if config.use_wandb:
            wandb.watch(model, criterion=model.loss_fn, log='all', idx=blueprint.id)

        print("Blueprint: {}\nModules: {}\nSample map: {}\n Species used: {}"
              .format(blueprint,
                      modules,
                      blueprint.best_module_sample_map,
                      list(set([x.species_id for x in blueprint.nodes.values()]))))
        print("Training model which scored: {} in evolution for {} epochs, with {}"
              .format(blueprint.accuracy, num_epochs, model.size()))

        accuracy = evaluate(model, num_epochs=num_epochs, fully_training=True)
        print('Achieved a final accuracy of: {}'.format(accuracy * 100))
        if config.use_wandb:
            wandb.log()
