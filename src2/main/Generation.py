"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import random
from typing import Optional

import src2.main.Singleton as Singleton
from src2.Genotype.CDN.PopulationInitializer import create_population, create_mr
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network
from src2.Genotype.NEAT.Population import Population
from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from src2.Genotype.NEAT.Operators.Speciators.NEATSpeciator import NEATSpeciator
from src2.main.ThreadManager import init_threads, reset_thread_name
from src2.Phenotype.NeuralNetwork.PhenotypeEvaluator import evaluate_blueprint
from src2.Configuration import config


class Generation:
    def __init__(self):
        Singleton.instance = self

        self.module_population: Optional[Population] = None
        self.blueprint_population: Optional[Population] = None
        self.da_population: Optional[Population] = None

        self.initialise_populations()
        self.module_population[0].visualize()

    def evaluate_blueprints(self):
        """Evaluates all blueprints multiple times."""
        # Multiplying and shuffling the blueprints so that config.evaluations number of blueprints is evaluated
        blueprints = list(self.blueprint_population) * config.evaluations
        print(blueprints)

        with ThreadPoolExecutor(max_workers=config.n_gpus, initializer=init_threads()) as ex:
            ex.map(evaluate_blueprint, blueprints)
        reset_thread_name()

    def initialise_populations(self):
        """Starts off the populations of a new evolutionary run"""
        self.module_population = Population(create_population(config.module_pop_size, ModuleNode, ModuleGenome),
                                            create_mr(), config.module_pop_size,
                                            NEATSpeciator(config.species_distance_thresh_mod_base,
                                                          config.n_module_species))
        self.blueprint_population = Population(create_population(config.bp_pop_size, BlueprintNode, BlueprintGenome),
                                               create_mr(), config.bp_pop_size, NEATSpeciator(1000, 1))
        # TODO DA pop

    def step(self):
        """
            Runs CDN for one generation. Calls the evaluation of all individuals. Prepares population objects for the
            next step.
        """
        print("step")
        self.evaluate_blueprints()
        print("evaluated blueprints")
        self.module_population.step()
        self.blueprint_population.step()


if __name__ == '__main__':
    Generation()
