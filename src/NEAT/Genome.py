import copy
import operator
import os
import random
import sys
from typing import Iterable

import graphviz
import networkx as nx
from data import DataManager
from networkx.algorithms.similarity import graph_edit_distance

from src.Config import Config, NeatProperties as Props
from src.NEAT.Gene import ConnectionGene, NodeGene, NodeType


class Genome:

    def __init__(self, connections: Iterable[ConnectionGene], nodes: Iterable[NodeGene]):
        self.rank = 0  # The order of this genome when ranked by fitness values
        self.uses = 0  # The numbers of times this genome is used
        self.fitness_values: list = [-(sys.maxsize - 1)]
        if Config.second_objective != '':
            self.fitness_values.append(
                sys.maxsize if Config.second_objective_comparator == operator.lt else -(sys.maxsize - 1))
        if Config.third_objective != '':
            self.fitness_values.append(
                sys.maxsize if Config.third_objective_comparator == operator.lt else -(sys.maxsize - 1))

        self._nodes = {}  # maps node id to node object
        for node in nodes:
            self.add_node(node)

        self._connected_nodes = set()  # set of (from,to)tuples
        self._connections = {}  # maps connection id to connection object
        for connection in connections:
            self.add_connection(connection, True)
        self.netx_graph = None

    def __gt__(self, other):
        return self.rank > other.rank

    def __lt__(self, other):
        return self.rank < other.rank

    def eq(self, other):
        if not isinstance(other, Genome):
            return False

        return self._nodes.keys() == other._nodes.keys() and self._connections.values() == other._connections.values()

    def get_disjoint_excess_genes(self, other):
        if not isinstance(other, Genome):
            raise TypeError('Expected type Genome, received type: ' + str(type(other)))

        return self._connections.keys() ^ other._connections.keys()

    def add_node(self, node):
        """Add node. Nodes must be added before their connections"""
        if node.id in self._nodes:
            raise Exception("Added node " + repr(node) + " already in genome " + repr(self))
        self._nodes[node.id] = node

    def add_connection(self, conn, ignore_height_exception=False):
        """Add connections. Nodes must be added before their connections"""
        if conn.id in self._connections:
            raise Exception("Added connection " + repr(conn) + " already in genome " + repr(self))
        if not (conn.from_node in self._nodes and conn.to_node in self._nodes):
            raise Exception("Trying to add connection. But nodes not in genome")
        if conn.from_node == conn.to_node:
            raise Exception("connection goes from node", conn.to_node, "to itself")
        if not ignore_height_exception and self._nodes[conn.from_node].height >= self._nodes[conn.to_node].height:
            raise Exception("Cannot add connections downwards, trying to connect heights " + repr(
                self._nodes[conn.from_node].height) + "->" + repr(self._nodes[conn.to_node].height))

        self._connected_nodes.add((conn.from_node, conn.to_node))
        self._connections[conn.id] = conn

    def report_fitness(self, fitnesses):
        """updates the fitnesses stored with a new given fitness"""
        if self.fitness_values is None or not self.fitness_values:
            self.fitness_values = [0 for _ in fitnesses]

        if Config.fitness_aggregation == "avg":
            for i, fitness in enumerate(fitnesses):
                if self.fitness_values[i] is None:
                    self.fitness_values[i] = 0
                self.fitness_values[i] = (self.fitness_values[i] * self.uses + fitness) / (self.uses + 1)
            self.uses += 1
        elif Config.fitness_aggregation == "max":
            for i, fitness in enumerate(fitnesses):
                if self.fitness_values[i] is None:
                    self.fitness_values[i] = 0
                self.fitness_values[i] = max(self.fitness_values[i], fitness)
        else:
            raise Exception("Unexpected fitness aggregation type: " + repr(Config.fitness_aggregation))

    def end_step(self, generation=None):
        """Resets all necessary values for next the generation"""
        self.uses = 0
        if self.fitness_values is not None:
            self.fitness_values = [0 for _ in self.fitness_values]

    def distance_to(self, other):
        """:returns distance from self to other"""
        return self.get_topological_distance(other)

    def get_netx_graph_form(self):
        """generates the netx graph used for graph edit distance"""
        if self.netx_graph is not None:
            return self.netx_graph
        G = nx.DiGraph()
        node_keys = []
        for node in self._nodes.values():
            node_keys.append((node.id, {'label': repr(node.id)}))
        G.add_nodes_from(node_keys)

        conn_keys = []
        for conn in self._connections.values():
            if Config.ignore_disabled_connections_for_topological_similarity and not conn.enabled():
                continue
            conn_keys.append((conn.to_node, conn.from_node, {'label': repr(conn.to_node) + "," + repr(conn.from_node)}))
        G.add_edges_from(conn_keys)

        self.netx_graph = G
        return G

    def get_topological_distance(self, other):
        """:returns either NEAT distance of graph edit distance from self to other (depending on Config option)"""
        if other == self:
            return 0

        num_disjoint = 0
        num_excess = 0

        self_max_conn_id = max(self._connections.keys())
        other_max_conn_id = max(other._connections.keys())
        smaller_id = min(self_max_conn_id, other_max_conn_id)

        for conn_id in self.get_disjoint_excess_genes(other):
            if Config.ignore_disabled_connections_for_topological_similarity:
                if conn_id in self._connections:
                    if not self._connections[conn_id].enabled():
                        continue
                else:
                    if not other._connections[conn_id].enabled():
                        continue

            if conn_id > smaller_id:
                num_excess += 1
            else:
                num_disjoint += 1

        neat_dist = (num_excess * Props.EXCESS_COEFFICIENT + num_disjoint * Props.DISJOINT_COEFFICIENT) / max(
            len(self._connections), len(other._connections))

        if Config.use_graph_edit_distance:
            match_func = lambda a, b: a['label'] == b['label']
            ged = graph_edit_distance(self.get_netx_graph_form(), other.get_netx_graph_form(), node_match=match_func,
                                      edge_match=match_func)
            if neat_dist > 0:
                pass

            return ged

        return neat_dist

    def mutate(self, mutation_record, attribute_magnitude=1, topological_magnitude=1, module_population=None, gen=-1):
        """each cdn genomme should override this method to do the specific mutations each type needs."""
        raise NotImplemented('Mutation should be called not in base class')

    def _mutate(self, mutation_record, add_node_chance, add_connection_chance, allow_connections_to_mutate=True,
                debug=False, attribute_magnitude=1, topological_magnitude=1):
        """the base mutation function which controls topological mutations and mutagen mutations"""
        if debug:
            print("Before mutation: ", self, "has branches;", self.has_branches())

        topology_changed = False
        if random.random() < add_node_chance:
            topology_changed = True
            random.choice(list(self._connections.values())).mutate_add_node(mutation_record, self)

        if random.random() < add_connection_chance:
            if debug:
                print("adding connection mutation")
            topology_changed = True
            mutated = False
            tries = 100
            # Keep trying the mutation until a valid one is found
            while mutated is False and tries > 0:
                mutated = self._mutate_add_connection(mutation_record,
                                                      random.choice(list(self._nodes.values())),
                                                      random.choice(list(self._nodes.values())))
                tries -= 1

        if allow_connections_to_mutate:
            if debug:
                print("connection change mutation")
            for connection in self._connections.values():
                orig_conn = copy.deepcopy(connection)
                mutated = connection.mutate(magnitude=topological_magnitude)
                topology_changed = topology_changed or mutated
                # If mutation made the genome invalid then undo it
                if mutated and not self.validate():
                    self._connections[orig_conn.id] = orig_conn

        for node in self._nodes.values():
            node.mutate(magnitude=attribute_magnitude)

        if topology_changed:
            self.calculate_heights()

        for mutagen in self.get_all_mutagens():
            mutagen.mutate(magnitude=attribute_magnitude)

        if debug:
            print("after mutation: ", self, "has branches;", self.has_branches())

        return self

    def get_all_mutagens(self):
        return []

    def _mutate_add_connection(self, mutation_record, node1, node2):
        """Adds a connection between to nodes if possible"""
        # Validation
        if node1.id == node2.id or node1.height == node2.height:
            return False

        from_node, to_node = (node1, node2) if (node1.height < node2.height) else (node2, node1)
        if (from_node.id, to_node.id) in self._connected_nodes:
            return False

        if from_node.node_type == NodeType.OUTPUT:
            raise Exception('Marked an output node as from node ' + repr(from_node))

        if to_node.node_type == NodeType.INPUT:
            raise Exception('Marked an input node as to node ' + repr(to_node))

        # Adding to global mutation dictionary
        mutation = (from_node.id, to_node.id)
        if mutation_record.exists(mutation):
            mutation_id = mutation_record.mutations[mutation]
        else:
            mutation_id = mutation_record.add_mutation(mutation)

        # Adding new mutation
        mutated_conn = ConnectionGene(mutation_id, from_node.id, to_node.id)
        self.add_connection(mutated_conn)

        return True

    def crossover(self, other):
        """performs neat style cross over between self and other as parents"""
        best = self if self < other else other
        worst = self if self > other else other

        child = type(best)([], [])

        for best_node in best._nodes.values():
            if best_node.id in worst._nodes:
                if Config.breed_mutagens and random.random() < Config.mutagen_breed_chance:
                    child_node = best_node.breed(worst._nodes[best_node.id])
                else:
                    child_node = copy.deepcopy(random.choice([best_node, worst._nodes[best_node.id]]))
            else:
                child_node = copy.deepcopy(best_node)

            child.add_node(child_node)

        for best_conn in best._connections.values():
            if self._nodes[best_conn.to_node].height <= self._nodes[best_conn.from_node].height:
                raise Exception("found connection in best parent which goes from height",
                                self._nodes[best_conn.from_node].height, "to", self._nodes[best_conn.to_node].height)
            if best_conn.id in worst._connections:
                new_connection = copy.deepcopy(random.choice([best_conn, worst._connections[best_conn.id]]))
            else:
                new_connection = copy.deepcopy(best_conn)

            child.add_connection(new_connection,
                                 ignore_height_exception=True)  # child heights not meaningful at this stage

        child.inherit(best)
        child.calculate_heights()
        return child

    def get_input_node(self):
        for node in self._nodes.values():
            if node.is_input_node():
                return node

        raise Exception('Genome:', self, 'could not find an input node')

    def get_output_node(self):
        for node in self._nodes.values():
            if node.is_output_node():
                return node

        raise Exception('Genome:', self, 'could not find an output node')

    def get_traversal_dictionary(self, exclude_disabled_connection=False, reverse=False):
        """
        :param exclude_disabled_connection: if true does not include nodes connected by disabled connections
        :param reverse: if true dictionary keys become 'to nodes' and values become 'from nodes'
        :returns: a mapping {node id -> list of connected node ids} for easy traversal
        """
        dictionary = {}

        for conn in self._connections.values():
            if not conn.enabled() and exclude_disabled_connection:
                continue

            key, value = (conn.from_node, conn.to_node) if not reverse else (conn.to_node, conn.from_node)

            if key not in dictionary:
                dictionary[key] = []

            dictionary[key].append(value)

        return dictionary

    def get_reachable_nodes(self, from_output):
        node_dict = self.get_traversal_dictionary(True, from_output)
        new_dict = {}

        start_id = self.get_input_node().id if not from_output else self.get_output_node().id
        self._get_reachable_nodes(node_dict, start_id, new_dict)
        return new_dict

    def _get_reachable_nodes(self, node_dict, curr_node, new_dict):
        if curr_node not in node_dict:
            return

        for node in node_dict[curr_node]:
            if curr_node not in new_dict:
                new_dict[curr_node] = []

            if node not in new_dict[curr_node]:  # don't add node id if already there
                new_dict[curr_node].append(node)

            self._get_reachable_nodes(node_dict, node, new_dict)

    def has_branches(self):
        """Checks if there are any paths that don't reach the output node that do not contain disabled connections"""
        traversal_dict = self.get_traversal_dictionary(exclude_disabled_connection=True)
        for children in traversal_dict.values():
            if len(children) > 1:
                return True
        return False

    def calculate_heights(self):
        for node in self._nodes.values():
            node.height = 0

        self._calculate_heights(self.get_input_node().id, 0, self.get_traversal_dictionary())

    def _calculate_heights(self, current_node_id, height, traversal_dictionary):
        """Calculates the heights of each node to make sure that no cycles can occur in the graph"""
        self._nodes[current_node_id].height = max(height, self._nodes[current_node_id].height)

        if self._nodes[current_node_id].is_output_node():
            return

        for child in traversal_dictionary[current_node_id]:
            """if any path doesn't reach the output - there has been an error"""
            self._calculate_heights(child, height + 1, traversal_dictionary)

    def validate(self):
        return self._validate_traversal(self.get_input_node().id, self.get_traversal_dictionary(True), set())

    def _validate_traversal(self, current_node_id, traversal_dictionary, nodes_visited):
        """Confirms that there is a path from input to output"""
        if self._nodes[current_node_id].is_output_node():
            return True
        if current_node_id in nodes_visited:
            return False

        nodes_visited.add(current_node_id)

        found_output = False
        if current_node_id in traversal_dictionary:
            for child in traversal_dictionary[current_node_id]:
                found_output = found_output or self._validate_traversal(child, traversal_dictionary, nodes_visited)

        return found_output

    def to_phenotype(self, Phenotype):
        """Converts self to a neural network"""
        phenotypes = {}

        root_node = None
        output_node = None
        for node in self._nodes.values():
            phenotypes[node.id] = Phenotype(node, self)
            if node.is_input_node():
                root_node = phenotypes[node.id]
            if node.is_output_node():
                output_node = phenotypes[node.id]

        for conn in self._connections.values():
            if not conn.enabled():
                continue

            if conn.from_node == conn.to_node:
                raise Exception("connection from and to the same node", conn.from_node)

            parent = phenotypes[conn.from_node]
            child = phenotypes[conn.to_node]

            parent.add_child(child)

        output_reaching_nodes = root_node.get_all_nodes_via_bottom_up(set())
        input_reaching_nodes = output_node.get_all_nodes_via_top_down(set())

        fully_connected_nodes = output_reaching_nodes & input_reaching_nodes

        if output_node not in fully_connected_nodes:
            raise Exception("output node not in fully connected nodes")

        for neat_node in self._nodes.values():
            graph_node = phenotypes[neat_node.id]

            if graph_node in fully_connected_nodes:
                continue

            if neat_node.node_type == NodeType.OUTPUT:
                raise Exception("Output node was not added to fully connected nodes")

            graph_node.severe_node()

        root_node.get_traversal_ids("_")
        return root_node

    def plot_tree_with_graphvis(self, title="", file="temp_g", view=False, graph=None, return_graph_obj=False,
                                node_prefix=""):

        file = os.path.join(DataManager.get_Graphs_folder(), file)

        if graph is None:
            graph = graphviz.Digraph(comment=title)

        for node in self._nodes.values():
            graph.node(node_prefix + str(node.id), node.get_node_name(), style="filled", fillcolor="white")

        for c in self._connections.values():
            if not c.enabled():
                continue
            graph.edge(node_prefix + repr(c.from_node), node_prefix + repr(c.to_node))

        graph.render(file, view=view)
        if return_graph_obj:
            return graph

    def get_fully_connected_nodes(self):
        """:returns only nodes that are connected to input and output node"""
        # Discard hanging nodes - i.e nodes that are only connected to either the input or output node
        node_map_from_input = self.get_reachable_nodes(False)
        node_map_from_output = self.get_reachable_nodes(True)
        # All non-hanging nodes excluding input and output node
        connected_nodes = node_map_from_input.keys() & node_map_from_output.keys()
        connected_nodes.add(self.get_input_node().id)  # Add input node
        connected_nodes.add(self.get_output_node().id)  # Add output node

        return connected_nodes

    def get_multi_input_nodes(self):
        # Find all nodes with multiple inputs
        multi_input_map = {}  # maps {node id: number of inputs}
        node_map = self.get_reachable_nodes(False)
        for node_id in self.get_fully_connected_nodes():
            num_inputs = sum(list(node_map.values()), []).count(node_id)
            if num_inputs > 1:
                multi_input_map[node_id] = num_inputs

        return multi_input_map

    def __repr__(self):
        return str(list(self._nodes.values())) + ' ' + str(list(self._connections.values()))
