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
        child = None

        child.calculate_heights()
        return child

    def get_input_node(self):
        pass

    def validate(self):
        found_output = self._validate_traversal(self.get_input_node(),self._get_traversal_dictionary(True))
        if not found_output:
            print("Error: could not find output node via bottom up traversal due to disabled connections")
            return False

        return True


    def _get_traversal_dictionary(self, exclude_disabled_connection=False):
        """:returns a mapping of from node id to a list of to node ids"""
        dictionary = {}

        for conn in self._connections:
            if not conn.enabled() and exclude_disabled_connection:
                continue
            if conn.from_node not in dictionary:
                dictionary[conn.from_node] = []

            dictionary[conn.from_node].append(conn.to_node)
        return dictionary

    def calculate_heights(self):
        for node in self._nodes:
            node.height = 0
            
        self._calculate_heights(self.get_input_node(), 0, self._get_traversal_dictionary())

    def _calculate_heights(self, current_node, height, traversal_dictionary):
        current_node.height = max(height, current_node.height)

        for child in traversal_dictionary[current_node]:
            """if any path doesn't reach the output - there has been an error"""
            self._calculate_heights(child, height + 1, traversal_dictionary)

    def _validate_traversal(self, current_node, traversal_dictionary):
        if current_node.is_output_node():
            return True

        found_output = False
        if current_node in traversal_dictionary:
            for child in traversal_dictionary[current_node]:
                found_output = found_output or self._validate_traversal(child, traversal_dictionary)

        return found_output


