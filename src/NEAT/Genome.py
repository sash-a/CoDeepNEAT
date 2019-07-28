from src.NEAT.Gene import ConnectionGene, NodeGene, NodeType
from typing import Iterable
import copy
import random
import sys
from src.Config import Config
import operator
import os
from data import DataManager
import graphviz


class Genome:

    def __init__(self, connections: Iterable[ConnectionGene], nodes: Iterable[NodeGene]):
        self.rank = 0  # The order of this genome when ranked by fitness values
        self.uses = 0  # The numbers of times this genome is used
        self.fitness_values: list = [-(sys.maxsize - 1)]
        if Config.second_objective != "":
            self.fitness_values.append(
                sys.maxsize if Config.second_objective_comparator == operator.lt else -(sys.maxsize - 1))
        if Config.third_objective != "":
            self.fitness_values.append(
                sys.maxsize if Config.third_objective_comparator == operator.lt else -(sys.maxsize - 1))

        self._nodes = {}
        for node in nodes:
            self.add_node(node)

        self._connected_nodes = set()  # set of (from,to)tuples
        self._connections = {}
        for connection in connections:
            self.add_connection(connection, True)


    def has_branches(self):
        traversal_dict = self._get_traversal_dictionary(exclude_disabled_connection=True)
        for children in traversal_dict.values():
            if len(children) >1 :
                return True
        return False

    def __gt__(self, other):
        return self.rank > other.rank

    def __lt__(self, other):
        return self.rank < other.rank

    def __repr__(self):
        return repr(list(self._connections.values()))

    def eq(self, other):
        if type(other) != type(self):
            return False

        return self._nodes.keys() == other._nodes.keys() and self._connections.values() == other._connections.values()

    def get_unique_genes(self, other):
        return set(self._connections.keys()) - set(other._connections.keys())

    def get_disjoint_genes(self, other):
        return set(self._connections.keys()) ^ set(other._connections.keys())

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
        #print("reporting fitness",fitnesses, "to genome:", type(self))
        if self.fitness_values is None or not self.fitness_values:
            self.fitness_values = [0 for _ in fitnesses]

        for i, fitness in enumerate(fitnesses):
            self.fitness_values[i] = (self.fitness_values[i] * self.uses + fitness) / (self.uses + 1)
        self.uses += 1

    def end_step(self):
        self.uses = 0
        if self.fitness_values is not None:
            self.fitness_values = [0 for _ in self.fitness_values]

    def distance_to(self, other):
        if other == self:
            return 0

        return len(self.get_disjoint_genes(other)) / max(len(self._connections), len(other._connections))

    def mutate(self, mutation_record):
        raise NotImplemented('Mutation should be called not in base class')

    def _mutate(self, mutation_record, add_node_chance, add_connection_chance, allow_connections_to_mutate=True, debug = False):
        if debug:
            print("before mutation: " , self, "has branches;",self.has_branches())

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
                mutated = connection.mutate()
                topology_changed = topology_changed or mutated
                # If mutation made the genome invalid then undo it
                if mutated and not self.validate():
                    self._connections[orig_conn.id] = orig_conn

        for node in self._nodes.values():
            node.mutate()

        if topology_changed:
            self.calculate_heights()

        for mutagen in self.get_all_mutagens():
            mutagen.mutate()

        if debug:
            print("after mutation: " , self, "has branches;",self.has_branches())

        return self

    def get_all_mutagens(self):
        return []

    def _mutate_add_connection(self, mutation_record, node1, node2):
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
        best = self if self < other else other
        worst = self if self > other else other

        child = type(best)([], [])

        for best_node in best._nodes.values():
            if best_node.id in worst._nodes:
                child.add_node(copy.deepcopy(random.choice([best_node, worst._nodes[best_node.id]])))
            else:
                child.add_node(copy.deepcopy(best_node))

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

        child.calculate_heights()
        return child

    def get_input_node(self):
        for node in self._nodes.values():
            if node.is_input_node():
                return node

        raise Exception('Genome:', self, 'could not find an output node')

    def _get_traversal_dictionary(self, exclude_disabled_connection=False):
        """:returns a mapping of from node id to a list of to node ids"""
        dictionary = {}

        for conn in self._connections.values():
            if not conn.enabled() and exclude_disabled_connection:
                continue
            if conn.from_node not in dictionary:
                dictionary[conn.from_node] = []

            dictionary[conn.from_node].append(conn.to_node)
        return dictionary

    def calculate_heights(self):
        for node in self._nodes.values():
            node.height = 0

        self._calculate_heights(self.get_input_node().id, 0, self._get_traversal_dictionary())

    def _calculate_heights(self, current_node_id, height, traversal_dictionary):
        self._nodes[current_node_id].height = max(height, self._nodes[current_node_id].height)

        if self._nodes[current_node_id].is_output_node():
            return

        for child in traversal_dictionary[current_node_id]:
            """if any path doesn't reach the output - there has been an error"""
            self._calculate_heights(child, height + 1, traversal_dictionary)

    def validate(self):
        return self._validate_traversal(self.get_input_node().id, self._get_traversal_dictionary(True), set())

    def _validate_traversal(self, current_node_id, traversal_dictionary, nodes_visited):
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
        phenotyes = {}

        root_node = None
        output_node = None
        for node in self._nodes.values():
            phenotyes[node.id] = Phenotype(node, self)
            if node.is_input_node():
                root_node = phenotyes[node.id]
            if node.is_output_node():
                output_node = phenotyes[node.id]

        for conn in self._connections.values():
            if not conn.enabled():
                continue

            if conn.from_node == conn.to_node:
                raise Exception("connection from and to the same node", conn.from_node)

            parent = phenotyes[conn.from_node]
            child = phenotyes[conn.to_node]

            parent.add_child(child)

        output_reaching_nodes = root_node.get_all_nodes_via_bottom_up(set())
        input_reaching_nodes = output_node.get_all_nodes_via_top_down(set())

        fully_connected_nodes = output_reaching_nodes & input_reaching_nodes

        if output_node not in fully_connected_nodes:
            raise Exception("output node not in fully connected nodes")

        for neat_node in self._nodes.values():
            graph_node = phenotyes[neat_node.id]

            if graph_node in fully_connected_nodes:
                continue

            if neat_node.node_type == NodeType.OUTPUT:
                raise Exception("Output node was not added to fully connected nodes")

            graph_node.severe_node()

        root_node.get_traversal_ids("_")
        return root_node

    def plot_tree_with_graphvis(self, title="", file="temp_g"):
        #print("genome_graph,1,2")
        file = os.path.join(DataManager.get_Graphs_folder(), file)
        print(file)

        graph = graphviz.Digraph(comment=title)

        for node in self._nodes.values():
            graph.node(str(node.id), node.get_node_name(),style="filled", fillcolor="blue")

        for c in self._connections.values():
            if not c.enabled():
                continue
            graph.edge(repr(c.from_node), repr(c.to_node))

        graph.render(file, view=Config.print_best_graphs)
