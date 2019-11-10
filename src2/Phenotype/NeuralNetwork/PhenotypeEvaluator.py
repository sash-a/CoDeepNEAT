from __future__ import annotations
from typing import TYPE_CHECKING

from src2.Configuration.Configuration import config
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
    inputs, targets = DataLoader.sample_data(config.get_device())

    model = Network(blueprint, self.module_population.species, list(inputs.size())).to(Config.get_device())
    s_train = time.time()
    new_acc = Validation.get_accuracy_estimate_for_network(n2, da_scheme=None, batch_size=Config.batch_size)


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
