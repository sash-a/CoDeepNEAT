from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, List

from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
    from src2.Genotype.NEAT import Population


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int]):
    """
    parses the blueprint into its phenotype NN
    handles the assignment of the single/multi obj finesses to the blueprint
    """
    model: Network = Network(blueprint, input_size)
    device = config.get_device()
    model.to(device)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable model params', model_size)

    if model_size > config.max_model_params:
        accuracy = 0
    else:
        accuracy = evaluate(model)

    blueprint.update_best_sample_map(model.sample_map, accuracy)
    blueprint.report_fitness([accuracy], module_sample_map=model.sample_map)
    print("Evaluation complete with accuracy:", accuracy)

    # if random.random() < 0.05:
    #     model.visualize()
    #     blueprint.visualize()

    return blueprint


def propagate_fitnesses_to_co_genomes(blueprint: BlueprintGenome):
    """
    passes the blueprints accuracy to the modules and da_individuals it used
    """


def assign_accuracy(phenotype: Network):
    """
    runs the NN training and testing to determine its test accuracy
    """
    pass


def assign_blueprint_complexity(phenotype: Network):
    """
    collects the complexities of the phenotypes and assigns them to the blueprint
    """
    pass


def assign_module_complexities(module_pop: Population):
    """
    collects the module complexities and assigns them
    """
