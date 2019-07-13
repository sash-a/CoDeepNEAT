from src.NEAT.Genome import Genome
from src.Module.ModuleNode import ModuleNode
from src.NEAT.Gene import ConnectionGene
from src.CoDeepNEAT.ModuleNEATNode import ModulenNEATNode
import src.Config.NeatProperties as Props

import copy


class ModuleGenome(Genome):

    def __init__(self, connections, nodes):
        super(ModuleGenome, self).__init__(connections, nodes)
        self.fitness_reports = 0
        self.module_node = None  # the module node created from this gene
        self.fitness = None

    def to_module(self):
        """
        returns the stored module_node of this gene, or generates and returns it if module_node is null
        :return: the module graph this individual represents
        """
        if self.module_node is not None:
            #print("module genome already has module - returning a copy")
            return copy.deepcopy(self.module_node)

        module = super().to_phenotype(ModuleNode)
        self.module_node = module
        return copy.deepcopy(module)

    def mutate(self, mutation_record):
        return super()._mutate(mutation_record, Props.MODULE_NODE_MUTATION_CHANCE, Props.MODULE_CONN_MUTATION_CHANCE)

    def report_fitness(self, *fitnesses):
        if self.fitness is None:
            self.fitness = [0 for _ in fitnesses]

        for i, fitness in enumerate(fitnesses):
            self.fitness[i] = (self.fitness[i] * self.fitness_reports + fitness) / (self.fitness_reports + 1)
        self.fitness_reports += 1

    def clear(self):
        self.fitness_reports = 0
        if self.fitness is not None:
            self.fitness = [0 for _ in self.fitness]
        self.module_node = None
