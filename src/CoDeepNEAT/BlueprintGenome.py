from src.NEAT.Genome import Genome
from src.Blueprint.Blueprint import BlueprintNode
from src.NEAT.Gene import ConnectionGene
from src.CoDeepNEAT.BlueprintNEATNode import BlueprintNEATNode

import src.Config.NeatProperties as Props


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections, nodes)
        self.modules_used = []  # holds ref to module individuals used - can multiple represent
        self.fitness = None

    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return super().to_phenotype(BlueprintNode)

    def mutate(self, mutation_record):
        return super()._mutate(mutation_record, Props.BP_NODE_MUTATION_CHANCE, Props.BP_CONN_MUTATION_CHANCE)

    def report_fitness(self, *fitnesses):
        if self.fitness is None:
            self.fitness = [0 for _ in fitnesses]

        for i, fitness in enumerate(fitnesses):
            self.fitness[i] = fitness

    def clear(self):
        self.modules_used = []

    def reset_number_of_module_species(self, num_module_species):
        for node in self._nodes.values():
            node.set_species_upper_bound(num_module_species)

    def __repr__(self):
        return '------------------Blueprint structure------------------\n' + \
               super().__repr__() + \
               '\n------------------Modules used------------------\n' + \
               repr([repr(module) for module in self.modules_used])
