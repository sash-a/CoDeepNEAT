from __future__ import annotations

from typing import TYPE_CHECKING

from src2.Configuration import config
from src.Validation import DataLoader
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
    from src2.Genotype.NEAT import Population


def evaluate_blueprint(blueprint: BlueprintGenome):
    """
    parses the blueprint into its phenotype NN
    handles the assignment of the single/multi obj finesses to the blueprint
    """
    print("evaling bps")
    inputs, targets = DataLoader.sample_data(config.get_device())
    print("loaded data")
    print('received bp: ', blueprint)
    model: Network = Network(blueprint, list(inputs.size())).to(config.get_device())
    print("created model")
    model.visualize()


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
