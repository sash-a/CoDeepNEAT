import random
from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NEATNode
import src.Config.NeatProperties as Props

from typing import Iterable


class Genome:

    def __init__(self, connections: Iterable[Connection], nodes: Iterable[NEATNode]):
        self.fitness = 0
        self.adjusted_fitness = 0

        self.nodes = []
        self.node_ids = set()
        for node in nodes:
            self.add_node(node)

        self.innov_nums = set()
        self.connections = []
        for connection in connections:
            self.add_connection(connection)

    def add_connection(self, new_connection):
        """Adds a new connection maintaining the order of the connections list"""
        if new_connection.innovation in self.innov_nums:
            return -1

        pos = 0
        for i, connection in enumerate(self.connections, 0):
            if new_connection.innovation < connection.innovation:
                break
            pos += 1

        self.connections.insert(pos, new_connection)
        self.innov_nums.add(new_connection.innovation)
        return pos

    def get_connection(self, innov):
        if innov not in self.innov_nums:
            return None

        for conn in self.connections:
            if conn.innovation == innov:
                return conn

        return None

    def add_node(self, new_node):
        """Adds a new node maintaining the order of the node list"""
        if new_node.id in self.node_ids:
            return -1

        pos = 0
        for i, node in enumerate(self.nodes, 0):
            if new_node.id < node.id:
                break
            pos += 1

        self.nodes.insert(pos, new_node)
        self.node_ids.add(new_node.id)
        return pos

    def get_node(self, id):
        if id not in self.node_ids:
            return None

        for node in self.nodes:
            if node.id == id:
                return node

    def distance_to(self, other_indv, c1=1, c2=1):
        n = max(len(self.connections), len(other_indv.connections))
        self_d, self_e = self.get_disjoint_excess(other_indv)
        other_d, other_e = other_indv.get_disjoint_excess(self)

        d = len(self_d) + len(other_d)
        e = len(self_e) + len(other_e)

        compatibility = (c1 * d + c2 * e) / n
        return compatibility

    # This is not as efficient but it is cleaner than a single pass
    def get_disjoint_excess(self, other):
        """Finds all of self's disjoint and excess genes when compared to other"""
        disjoint = []
        excess = []

        other_max_innov = other.connections[-1].innovation

        for connection in self.connections:
            if connection.innovation not in other.innov_nums:
                if connection.innovation > other_max_innov:
                    excess.append(connection)
                else:
                    disjoint.append(connection)

        return disjoint, excess

    def _mutate_add_node(self, conn: Connection, mutations: dict, innov: int, node_id: int,  MutationType=NEATNode):
        conn.enabled.set_value(False)

        mutated_node = MutationType(node_id + 1, conn.from_node.midpoint(conn.to_node))

        mutated_from_conn = Connection(conn.from_node, mutated_node, innovation=innov + 1)
        mutated_to_conn = Connection(mutated_node, conn.to_node, innovation=innov + 2)

        innov += 2
        node_id += 1

        if conn.innovation in mutations:
            prev_id = mutations[conn.innovation]
            from_innov = mutations[(conn.from_node.id, prev_id)]
            to_innov = mutations[(prev_id, conn.to_node.id)]

            mutated_node.id = prev_id
            mutated_from_conn.innovation = from_innov
            mutated_to_conn.innovation = to_innov

            innov -= 2
            node_id -= 1
        else:
            # Tracking the added node
            mutations[conn.innovation] = node_id
            # Tracking the added connections
            mutations[(conn.from_node.id, node_id)] = mutated_from_conn.innovation
            mutations[(node_id, conn.to_node.id)] = mutated_to_conn.innovation

        self.add_connection(mutated_from_conn)
        self.add_connection(mutated_to_conn)
        self.add_node(mutated_node)

        return innov, node_id

    def _mutate_add_connection(self, node1: NEATNode, node2: NEATNode, mutations: dict, innov: int):
        from_node, to_node = (node1, node2) if node1.x < node2.x else (node2, node1)
        mutated_conn = Connection(from_node, to_node)

        # Make sure nodes aren't equal and there isn't already a connection between them
        if node1.id == node2.id or \
                mutated_conn in self.connections or \
                Connection(to_node, from_node) in self.connections:
            return None

        # Check if the connection exist somewhere else
        possible_mutation = (from_node.id, to_node.id)
        if possible_mutation not in mutations:
            innov += 1
            mutated_conn.innovation = innov
            mutations[possible_mutation] = innov
        else:
            mutated_conn.innovation = mutations[possible_mutation]

        self.add_connection(mutated_conn)

        return innov

    def mutate(self, mutations: dict, innov: int, node_id: int, node_chance=0.1, conn_chance=0.15):

        if random.random() < node_chance:  # Add a new node
            innov, node_id = self._mutate_add_node(random.choice(self.connections), mutations, innov, node_id)

        if random.random() < conn_chance:  # Add a new connection
            # Try until find acceptable nodes
            for _ in range(Props.MUTATION_TRIES):
                outcome = self._mutate_add_connection(
                    random.choice(self.nodes),
                    random.choice(self.nodes),
                    mutations, innov)

                # Found acceptable nodes
                if outcome is not None:
                    innov = outcome
                    break

        for mutagen in self.get_all_mutagens():
            mutagen.mutate()

        for node in self.nodes:
            node.mutate()

        for connection in self.connections:
            connection.mutate()

        return innov, node_id

    def to_phenotype(self, Phenotype):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """

        graph_node_map = {}
        root_node = None

        # initialises nodes and maps them to their genes
        for neat_node in self.nodes:
            graph_node_map[neat_node.id] = Phenotype(neat_node, self)
            if neat_node.is_input_node():
                root_node = graph_node_map[neat_node.id]

        # connects the blueprint nodes as indicated by the genome
        for connection in self.connections:
            if not connection.enabled.get_value():
                continue
            try:
                parent = graph_node_map[connection.from_node.id]
                child = graph_node_map[connection.to_node.id]

                parent.add_child(child)
            except KeyError:
                pass

        output_reaching_nodes = root_node.get_all_nodes_via_bottom_up(set())
        input_reaching_nodes = root_node.get_output_node().get_all_nodes_via_top_down(set())

        fully_connected_nodes = output_reaching_nodes.intersection(input_reaching_nodes)
        for neat_node in self.nodes:
            graph_node = graph_node_map[neat_node.id]
            if graph_node in fully_connected_nodes:
                continue
            graph_node.severe_node()

        root_node.get_traversal_ids("_")
        return root_node

    def __repr__(self):
        return str(self.connections)

    def get_all_mutagens(self):
        return []
