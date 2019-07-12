from src.NEAT2.Gene import ConnectionGene, NodeGene, NodeType
from typing import Iterable
import copy
import random
from src.Config import NeatProperties as Props


class Genome:

    def __init__(self, connections: Iterable[ConnectionGene], nodes: Iterable[NodeGene], species):
        self.rank = 0  # accuracy is single OBJ else rank
        self.fitness_values: list = []

        self._nodes = {}
        for node in nodes:
            self.add_node(node)

        self._connections = {}
        for connection in connections:
            self.add_connection(connection)

        self._connected_nodes = set()  # set of (from,to)tuples

        self.species = species

    def __gt__(self, other):
        return self.rank > other.rank

    def __lt__(self, other):
        return self.rank < other.rank

    def get_unique_genes(self, other):
        return set(self._connections.keys()) - set(other._connections.keys())

    def get_disjoint_genes(self, other):
        return set(self._connections.keys()) ^ set(other._connections.keys())

    def add_node(self, node):
        """Add nodes"""
        if node.id in self._nodes:
            raise Exception("Added node " + repr(node) + " already in genome " + repr(self))
        self._nodes[node.id] = node

    def add_connection(self, conn):
        """
            Add connections
            Nodes must be added before their connections
        """
        if conn.id in self._connections:
            raise Exception("Added connection " + repr(conn) + " already in genome " + repr(self))
        if not (conn.from_node in self._nodes and conn.to_node in self._nodes):
            raise Exception("Trying to add connection. But nodes not in genome")
        if self._nodes[conn.from_node].height >= self._nodes[conn.to_node].height:
            raise Exception("Cannot add connections downwards, trying to connect heights " + repr(
                self._nodes[conn.from_node].height) + "->" + repr(self._nodes[conn.to_node].height))

        self._connected_nodes.add((conn.from_node, conn.to_node))
        self._connections[conn.id] = conn

    def distance_to(self, other):
        if other == self:
            return 0

        return len(self.get_disjoint_genes(other)) / max(len(self._connections), len(other._connections))

    def mutate(self, mutation_record):
        raise NotImplemented('Mutation should be called not in base class')

    def _mutate(self, mutation_record, add_node_chance, add_connection_chance):
        topology_changed = False
        if random.random() < add_node_chance:
            topology_changed = True
            random.choice(self._connections.values()).mutate_add_node()

        if random.random() < add_connection_chance:
            topology_changed = True
            mutated = False
            while mutated is False:
                mutated = self._mutate_add_connection(mutation_record,
                                                      random.choice(self._nodes.values()),
                                                      random.choice(self._nodes.values()))

        # TODO mutate other stuff (non-topological)

        if topology_changed:
            self.calculate_heights()

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
            if best_conn.id in worst._connections:
                child.add_connection(copy.deepcopy(random.choice([best_conn, worst._connections[best_conn.id]])))
            else:
                child.add_connection(copy.deepcopy(best_conn))

        child.calculate_heights()
        return child

    def get_input_node(self):
        for node in self._nodes:
            if node.is_input_node():
                return node

        raise Exception('Genome:', self, 'could not find an output node')

    def validate(self):
        found_output = self._validate_traversal(self.get_input_node(), self._get_traversal_dictionary(True))
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

        sampled_trailing_node = root_node.get_output_node()
        while not sampled_trailing_node == output_node:
            # print("sampled a false output node:", sampled_trailing_node, "real output node:", output_node)
            sampled_trailing_node.severe_node()
            sampled_trailing_node = root_node.get_output_node()
            if sampled_trailing_node == root_node:
                print(self)
                raise Exception("root node is output node - num children:", len(root_node.children))

        output_reaching_nodes = root_node.get_all_nodes_via_bottom_up(set())
        input_reaching_nodes = output_node.get_all_nodes_via_top_down(set())

        fully_connected_nodes = output_reaching_nodes.intersection(input_reaching_nodes)

        if not output_node in fully_connected_nodes:
            raise Exception("output node not in fully connected nodes")

        for neat_node in self._nodes.values():
            graph_node = phenotyes[neat_node.id]
            if graph_node in fully_connected_nodes:
                continue
            if neat_node.node_type == NodeType.OUTPUT:
                raise Exception("severing the neat output node, is_graph_output_node:", graph_node.is_output_node())
            graph_node.severe_node()

        root_node.get_traversal_ids("_")
        return root_node
