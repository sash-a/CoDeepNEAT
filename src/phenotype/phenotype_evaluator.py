from __future__ import annotations

import torch.multiprocessing as mp
from typing import TYPE_CHECKING, List

from src.configuration import config
from src.phenotype.neural_network.evaluator.evaluator import evaluate
from src.phenotype.neural_network.neural_network import Network

import src.main.singleton as singleton

if TYPE_CHECKING:
    from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome


def evaluate_blueprints(blueprint_q: mp.Queue,
                        input_size: List[int],
                        num_epochs: int = config.epochs_in_evolution) -> List[BlueprintGenome]:
    """
    Consumes blueprints off the blueprints queue, evaluates them and adds them back to the queue if all of their
    evaluations have not been completed for the current generation. If all their evaluations have been completed, add
    them to the completed_blueprints list.

    :param blueprint_q:
    :param input_size:
    :param num_epochs:
    :return:
    """
    completed_blueprints: List[BlueprintGenome] = []
    while blueprint_q.qsize() != 0:
        blueprint = blueprint_q.get()

        blueprint = evaluate_blueprint(blueprint, input_size, num_epochs)
        if blueprint.n_evaluations == config.n_evals_per_bp:
            completed_blueprints.append(blueprint)
        else:
            blueprint_q.put(blueprint)

    return completed_blueprints


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int], num_epochs) -> BlueprintGenome:
    """
    Parses the blueprint into its phenotype NN
    Handles the assignment of the single/multi obj finesses to the blueprint in parallel
    """
    device = config.get_device()
    model: Network = Network(blueprint, input_size).to(device)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_size > config.max_model_params:
        print("dropped model which was too large:", model_size, "params")
        accuracy = 0
    else:
        accuracy = evaluate(model, n_epochs=num_epochs)

    blueprint.update_best_sample_map(model.sample_map, accuracy)
    fitness = [accuracy, model_size]
    blueprint.report_fitness(fitness)
    parse_number = blueprint.n_evaluations

    print("Blueprint - {:^5} - accuracy: {:05.2f}% (proc {})"
          .format(blueprint.id, accuracy * 100, mp.current_process().name))

    if config.plot_every_genotype:
        blueprint.visualize(parse_number=parse_number,
                            prefix="g" + str(singleton.instance.generation_number) + "_" + str(blueprint.id))

    if config.plot_every_phenotype:
        model.visualize(parse_number=parse_number,
                        prefix="g" + str(singleton.instance.generation_number) + "_" + str(blueprint.id))

    return blueprint
