"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""
from concurrent.futures import ThreadPoolExecutor
import random

from src2.Genotype.NEAT.Population import Population
from src2.main.ThreadManager import init_threads, reset_thread_name
from src2.Phenotype.PhenotypeEvaluator import evaluate_blueprint
from src2.Configuration import config


class Generation:
    def __init__(self):
        self.module_population: Population = None
        self.blueprint_population: Population = None
        self.da_population: Population = None

    def evaluate_blueprints(self):
        """Evaluates all blueprints multiple times."""
        # Multiplying and shuffling the blueprints so that config.evaluations number of blueprints is evaluated
        blueprints = list(self.blueprint_population) * config.evaluations / len(self.blueprint_population)
        random.shuffle(blueprints)
        blueprints = blueprints[:config.evaluations]

        with ThreadPoolExecutor(max_workers=config.n_gpus, initializer=init_threads()) as ex:
            ex.map(evaluate_blueprint, blueprints)
        reset_thread_name()

    def initialise_populations(self):
        """Starts off the populations of a new evolutionary run"""
        pass

    def step(self):
        """
            Runs CDN for one generation
            calls the evaluation of all individuals
            prepares population objects for the next step
        """
        self.evaluate_blueprints()
