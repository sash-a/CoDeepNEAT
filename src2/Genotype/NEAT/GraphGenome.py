from __future__ import annotations

import time
from typing import Dict, TYPE_CHECKING, List, Union, AbstractSet, Set

from tarjan import tarjan

from src2.Configuration import config
from src2.Genotype.NEAT.MutationRecord import MutationRecords

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome
    from src2.Genotype.NEAT.Connection import Connection
    from src2.Genotype.NEAT.Node import Node
    from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
    from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode


class GraphGenome:
    def __init__(self, nodes: List[Node], connections: List[Connection]):
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

    def get_disjoint_excess_connections(self, other: Genome) -> AbstractSet[int]:
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

    def n_mutations_on_connection(self, record: MutationRecords, conn_id: int):
        """finds the next usable node ID given the mutation record and the nodes in the genome"""
        count = 0
        found_next_id = False
        while not found_next_id:
            mutation = (conn_id, count)
            if record.exists(mutation, False):
                node_id = record.node_mutations[mutation]
                if node_id in self.nodes:
                    # If mutation has already occurred in this genome then continue searching for a valid node id
                    count += 1
                    continue

            found_next_id = True

        # if count > 0:
        #     print("using count =", count)
        return count

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
        return False

    def get_input_node(self) -> Union[Node, ModuleNode, BlueprintNode]:
        return [node for node in self.nodes.values() if node.is_input_node()][0]

    def get_output_node(self) -> Union[Node, ModuleNode, BlueprintNode]:
        return [node for node in self.nodes.values() if node.is_output_node()][0]

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
        self._get_reachable_nodes(traversal_dict, start_id, reachable_dict)

        return reachable_dict

    def _get_reachable_nodes(self, traversal_dict, from_node, reachable_dict):
        """Recursive method for get reachable nodes"""
        if from_node not in traversal_dict:
            return

        if from_node not in reachable_dict:
            reachable_dict[from_node] = []

        for to_node in traversal_dict[from_node]:
            if to_node not in reachable_dict[from_node]:  # don't add node id if already there
                reachable_dict[from_node].append(to_node)

            self._get_reachable_nodes(traversal_dict, to_node, reachable_dict)

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

    def get_multi_input_nodes(self) -> Dict[int, int]:
        """Find all nodes with multiple inputs"""
        multi_input_map = {}  # maps {node id: number of inputs}
        node_map = self.get_reachable_nodes(False)
        for node_id in self.get_fully_connected_node_ids():
            num_inputs = sum(list(node_map.values()), []).count(node_id)
            if num_inputs > 1:
                multi_input_map[node_id] = num_inputs

        return multi_input_map
