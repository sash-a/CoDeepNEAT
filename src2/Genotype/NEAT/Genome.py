import copy
import sys
from typing import Dict, List, AbstractSet

from src2.Configuration import config
from src2.Genotype.Mutagen.Mutagen import Mutagen
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Node import Node


class Genome:
    _id_counter = 0

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.id = Genome._id_counter
        Genome._id_counter += 1

        self.rank = 0  # The order of this genome when ranked by fitness values
        self.uses = 0  # The numbers of times this genome is used
        self.fitness_values: List[int] = [-(sys.maxsize - 1)]

        # nodes and connections map from gene id -> gene object
        self.nodes: Dict[int, Node] = {}
        self.connections: Dict[int, Connection] = {}

        # connected nodes is stored to quickly tell if a connection is already in the genome
        self.connected_nodes = set()  # set of (from,to)tuples

        for node in nodes:
            self.nodes[node.id] = node

        for con in connections:
            self.connections[con.id] = con

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def get_disjoint_excess_connections(self, other) -> AbstractSet[int]:
        if not isinstance(other, Genome):
            raise TypeError('Expected type Genome, received type: ' + str(type(other)))

        return self.connections.keys() ^ other.connections.keys()

    def add_node(self, node):
        """Add node. Nodes must be added before their connections"""
        if node.id in self.nodes:
            raise Exception("Added node " + repr(node) + " already in genome " + repr(self))
        self.nodes[node.id] = node

    def add_connection(self, conn):
        """Add connections. Nodes must be added before their connections"""
        if conn.id in self.connections:
            raise Exception("Added connection " + repr(conn) + " already in genome " + repr(self))
        if not (conn.from_node in self.nodes and conn.to_node in self.nodes):
            raise Exception("Trying to add connection. But nodes not in genome")
        if conn.from_node == conn.to_node:
            raise Exception("connection goes from node", conn.to_node, "to itself")

        self.connected_nodes.add((conn.from_node, conn.to_node))
        self.connections[conn.id] = conn

    def report_fitness(self, fitnesses):
        """updates the fitnesses stored with a new given fitness"""
        if self.fitness_values is None or not self.fitness_values:
            self.fitness_values = [0 for _ in fitnesses]

        if config.fitness_aggregation == 'avg':
            for i, fitness in enumerate(fitnesses):
                if self.fitness_values[i] is None:
                    self.fitness_values[i] = 0
                self.fitness_values[i] = (self.fitness_values[i] * self.uses + fitness) / (self.uses + 1)
            self.uses += 1
        elif config.fitness_aggregation == 'max':
            for i, fitness in enumerate(fitnesses):
                if self.fitness_values[i] is None:
                    self.fitness_values[i] = 0
                self.fitness_values[i] = max(self.fitness_values[i], fitness)
        else:
            raise Exception("Unexpected fitness aggregation type: " + repr(config.fitness_aggregation))

    def end_step(self):
        """Resets all necessary values for next the generation"""
        self.uses = 0
        if self.fitness_values is not None:
            self.fitness_values = [0 for _ in self.fitness_values]

    def get_traversal_dictionary(self, exclude_disabled_connection=False, reverse=False) -> Dict[int, List[int]]:
        """
        :param exclude_disabled_connection: if true does not include nodes connected by disabled connections
        :param reverse: if true dictionary keys become 'to nodes' and values become 'from nodes'
        :returns: a mapping {node id -> list of connected node ids} for easy traversal
        """
        dictionary = {}
        for conn in self.connections.values():
            if not conn.enabled() and exclude_disabled_connection:
                continue

            key, value = (conn.from_node_id, conn.to_node_id) if not reverse else (conn.to_node_id, conn.from_node_id)

            if key not in dictionary:
                dictionary[key] = []

            dictionary[key].append(value)

        return dictionary

    def has_cycle(self) -> bool:
        visited_ids = set()
        traversal_dict = self.get_traversal_dictionary(True)
        return self._has_cycle(self.get_input_node().id, traversal_dict, visited_ids)

    def _has_cycle(self, current_node_id, traversal_dict, visited_set: set) -> bool:
        # TODO test and test performance
        if current_node_id in visited_set:
            return True

        if current_node_id not in traversal_dict:
            return False

        children = traversal_dict[current_node_id]
        visited_set.add(current_node_id)

        if len(children) == 1:
            return self._has_cycle(children[0], traversal_dict, visited_set)
        else:
            has_cycle = False
            for child in children:
                branch_visited_set = copy.deepcopy(visited_set)
                has_cycle = self._has_cycle(child, traversal_dict, branch_visited_set)
            return has_cycle

    def get_input_node(self) -> Node:
        return [node for node in self.nodes.values() if node.is_input_node()][0]

    def get_output_node(self) -> Node:
        return [node for node in self.nodes.values() if node.is_output_node()][0]

    def distance_to(self, other) -> float:
        return self.get_topological_distance(other)

    def get_topological_distance(self, other) -> float:
        """:returns normal NEAT distance to"""
        if other == self:
            return 0

        num_disjoint = 0
        num_excess = 0

        self_max_conn_id = max(self.connections.keys())
        other_max_conn_id = max(other.connections.keys())
        smaller_id = min(self_max_conn_id, other_max_conn_id)

        for conn_id in self.get_disjoint_excess_connections(other):
            if conn_id > smaller_id:
                num_excess += 1
            else:
                num_disjoint += 1

        neat_dist = (num_excess * config.excess_coefficient + num_disjoint * config.disjoint_coefficient) / max(
            len(self.connections), len(other.connections))

        return neat_dist

    def has_branches(self) -> bool:
        """Checks if there are any paths that don't reach the output node that do not contain disabled connections"""
        traversal_dict = self.get_traversal_dictionary(exclude_disabled_connection=True)
        for children in traversal_dict.values():
            if len(children) > 1:
                return True
        return False

    def validate(self) -> bool:
        connected = self._validate_traversal(self.get_input_node().id, self.get_traversal_dictionary(True), set())
        return connected and not self.has_cycle()

    def _validate_traversal(self, current_node_id, traversal_dictionary, nodes_visited) -> bool:
        """Confirms that there is a path from input to output"""
        if self.nodes[current_node_id].is_output_node():
            return True
        if current_node_id in nodes_visited:
            return False

        nodes_visited.add(current_node_id)

        found_output = False
        if current_node_id in traversal_dictionary:
            for child in traversal_dictionary[current_node_id]:
                found_output = found_output or self._validate_traversal(child, traversal_dictionary, nodes_visited)

        return found_output

    def to_phenotype(self):
        pass

    def get_all_mutagens(self) -> List[Mutagen]:
        return []
