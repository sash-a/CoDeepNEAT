"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import random

import src2.main.Singleton as S
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Node import NodeType
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network
from test.StaticGenomes import get_small_tri_genome
from src2.Genotype.NEAT.Population import Population
from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from src2.Genotype.NEAT.Operators.Speciators.NEATSpeciator import NEATSpeciator
from src2.main.ThreadManager import init_threads, reset_thread_name
from src2.Phenotype.NeuralNetwork.PhenotypeEvaluator import evaluate_blueprint
from src2.Configuration.Configuration import config
from src2.Visualisation.GenomeVisualiser import get_graph_of
from src2.Visualisation import GenomeVisualiser

instance: Generation


class Generation:

    def __init__(self):
        global instance
        self.module_population: Population = None
        self.blueprint_population: Population = None
        self.da_population: Population = None
        instance = self
        S.instance = self

        mod, modmr = get_small_tri_genome(ModuleGenome, ModuleNode)
        bp, bpmr = get_small_tri_genome(BlueprintGenome, BlueprintNode)
        spctr = NEATSpeciator(3, 1)
        for node in bp.nodes.values():
            node.species_id = 0
        self.module_population = Population([mod], modmr, 1, spctr)
        self.blueprint_population = Population([bp], bpmr, 1, spctr)

        import src.Validation.DataLoader as DL

        x, target = DL.sample_data(config.get_device(), 2)
        n = Network(bp, list(x.shape))
        n.visualize()
        # print(n)
        # # get_graph_of(bp).view()
        # GenomeVisualiser.visualise_blueprint_genome(bp)

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


if __name__ == '__main__':
    Generation()
