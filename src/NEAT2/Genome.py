from src.NEAT2.Gene import ConnectionGene, NodeGene
from typing import Iterable


class Genome:

    def __init__(self, connections: Iterable[ConnectionGene], nodes: Iterable[NodeGene], species):
        self.rank = 0  # accuracy is single OBJ else rank
        self.fitness_values: list = []

        self._nodes = []
        self._node_ids = set()
        for node in nodes:
            self.add_node(node)

        self._conn_innovations = set()
        self._connections = []
        for connection in connections:
            self.add_connection(connection)

        self.species = species

    def get_unique_genes(self, other):
        return self._conn_innovations - other._conn_innovations

    def get_disjoint_genes(self, other):
        return self._conn_innovations ^ other._conn_innovations

    def add_node(self, node):
        """Add nodes maintaining order"""
        self._nodes.append(node)
        self._node_ids.add(node.id)

    def add_connection(self, conn):
        """Add connections maintaining order"""
        self._connections.append(conn)
        self._conn_innovations.add(conn.id)

    def distance_to(self, other):
        if other == self:
            return 0

        return len(self.get_disjoint_genes(other)) / max(len(self._conn_innovations), len(other._conn_innovations))

    def mutate(self):
        pass

    def crossover(self, other):
        pass

    def get_input_node(self):
        pass

    def validate(self):
        try:
            self.calculate_heights(self.get_input_node())
            return True
        except Exception as e:
            print("Error: Invalid genome:", e)
            return False

    def calculate_heights(self, current_node):
        pass
