from src.NEAT2.Gene import ConnectionGene, NodeGene
from typing import Iterable


class Chromosome:
    def __init__(self, connections: Iterable[ConnectionGene], nodes: Iterable[NodeGene], species):
        self.fitness = 0
        self.adjusted_fitness = 0

        self.uses = 0

        self._nodes = []
        self._node_ids = set()
        for node in nodes:
            self.add_node(node)

        self._conn_innovations = set()
        self._connections = []
        for connection in connections:
            self.add_connection(connection)

        self.species = species

    def add_node(self, node):
        """Add nodes maintaining order"""
        pass

    def add_connection(self, conn):
        """Add connections maintaining order"""
        pass

    def distance_to(self, other):
        pass

    def mutate(self):
        pass

    def crossover(self, other):
        pass
