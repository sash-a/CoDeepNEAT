from __future__ import annotations

import sys
from typing import Dict, List, AbstractSet, Union, TYPE_CHECKING, Set

from tarjan import tarjan

from src2.Configuration import config
from src2.Genotype.Mutagen.Mutagen import Mutagen

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Connection import Connection
    from src2.Genotype.NEAT.Node import Node
    from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
    from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode


class Genome:
    _id_counter = 0

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.id = Genome._id_counter
        Genome._id_counter += 1

        self.rank = 0  # The order of this genome when ranked by fitness values
        self.uses = 0  # The numbers of times this genome is used
        self.fitness_values: List[int] = [-(sys.maxsize - 1)]

        # nodes and connections map from gene id -> gene object
        self.nodes: Dict[int, Union[Node, BlueprintNode, ModuleNode]] = {}
        self.connections: Dict[int, Connection] = {}

        # connected nodes is stored to quickly tell if a connection is already in the genome
        self.connected_nodes = set()  # set of (from,to)tuples

        for node in nodes:
            self.nodes[node.id] = node

        for con in connections:
            self.connections[con.id] = con
            self.connected_nodes.add((con.from_node_id, con.to_node_id))

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def get_disjoint_excess_connections(self, other: Genome) -> AbstractSet[int]:
        if not isinstance(other, Genome):
            raise TypeError('Expected type Genome, received type: ' + str(type(other)))

        return self.connections.keys() ^ other.connections.keys()

    def add_node(self, node: Node):
        """Add node. Nodes must be added before their connections"""
        if node.id in self.nodes:
            raise Exception("Added node " + repr(node) + " already in genome " + repr(self))
        self.nodes[node.id] = node

    def add_connection(self, conn: Connection):
        """Add connections. Nodes must be added before their connections"""
        if conn.id in self.connections:
            raise Exception("Added connection " + repr(conn) + " already in genome " + repr(self))
        if not (conn.from_node_id in self.nodes and conn.to_node_id in self.nodes):
            raise Exception("Trying to add connection. But nodes not in genome")
        if conn.from_node_id == conn.to_node_id:
            raise Exception("connection goes from node", conn.to_node_id, "to itself")

        self.connected_nodes.add((conn.from_node_id, conn.to_node_id))
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
        cycles = tarjan(self.get_traversal_dictionary(exclude_disabled_connection=True))

        for cycle in cycles:
            if len(cycle) > 1:
                return True
        # print('no cycle')
        # print(cycles)
        # print(self.get_traversal_dictionary(exclude_disabled_connection=True))
        return False

    def get_input_node(self) -> Union[Node, ModuleNode, BlueprintNode]:
        return [node for node in self.nodes.values() if node.is_input_node()][0]

    def get_output_node(self) -> Union[Node, ModuleNode, BlueprintNode]:
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

    def get_fully_connected_connections(self) -> Set[Connection]:
        connected_nodes = self.get_fully_connected_node_ids()
        included_connections = set()
        for connection in self.connections.values():
            if connection.to_node_id in connected_nodes and connection.from_node_id in connected_nodes:
                included_connections.add(connection)

        return included_connections

    def get_reachable_nodes(self, from_output_to_input: bool) -> Dict[int, List[int]]:
        traversal_dict = self.get_traversal_dictionary(True, from_output_to_input)
        reachable_dict = {}

        start_id = self.get_input_node().id if not from_output_to_input else self.get_output_node().id
        _get_reachable_nodes(traversal_dict, start_id, reachable_dict)
        return reachable_dict

    def get_fully_connected_node_ids(self) -> Set[int]:
        """:returns only nodes that are connected to input and output node"""
        # Discard hanging nodes - i.e nodes that are only connected to either the input or output node
        node_map_from_input = self.get_reachable_nodes(from_output_to_input=False)
        node_map_from_output = self.get_reachable_nodes(from_output_to_input=True)
        # All non-hanging nodes excluding input and output node
        connected_nodes: Set[int] = set(node_map_from_input.keys() & node_map_from_output.keys())
        connected_nodes.add(self.get_input_node().id)  # Add input node
        connected_nodes.add(self.get_output_node().id)  # Add output node

        return connected_nodes


def _get_reachable_nodes(traversal_dict, from_node, reachable_dict):
    if from_node not in traversal_dict:
        return

    for to_node in traversal_dict[from_node]:
        if from_node not in reachable_dict:
            reachable_dict[from_node] = []

        if to_node not in reachable_dict[from_node]:  # don't add node id if already there
            reachable_dict[from_node].append(to_node)

        _get_reachable_nodes(traversal_dict, to_node, reachable_dict)
