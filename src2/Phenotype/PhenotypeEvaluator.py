from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.NEAT import Population
from src2.Phenotype import NeuralNetwork


def evaluate_blueprint(blueprint: BlueprintGenome):
    """
    parses the blueprint into its phenotype NN
    handles the assignment of the single/multi obj finesses to the blueprint
    """
    pass


def propagate_fitnesses_to_co_genomes(blueprint: BlueprintGenome):
    """
    passes the blueprints accuracy to the modules and da_individuals it used
    """


def assign_accuracy(phenotype: NeuralNetwork):
    """
    runs the NN training and testing to determine its test accuracy
    """
    pass


def assign_blueprint_complexity(phenotype: NeuralNetwork):
    """
    collects the complexities of the phenotypes and assigns them to the blueprint
    """
    pass


def assign_module_complexities(module_pop: Population):
    """
    collects the module complexities and assigns them
    """
