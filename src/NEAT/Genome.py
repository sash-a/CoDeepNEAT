import random
from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NEATNode, NodeType
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
        self.defective = False

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

    def _mutate_add_node(self, conn: Connection, mutations: dict, innov: int, node_id: int, MutationType=NEATNode):
        conn.enabled.set_value(False)

        mutated_node = MutationType(node_id + 1, conn.from_node.midpoint(conn.to_node))

        # TODO deepcopy?
        mutated_from_conn = Connection(conn.from_node, mutated_node, innovation=innov + 1)
        mutated_to_conn = Connection(mutated_node, conn.to_node, innovation=innov + 2)

        if conn.innovation in mutations:
            prev_id = mutations[conn.innovation]
            from_innov = mutations[(conn.from_node.id, prev_id)]
            to_innov = mutations[(prev_id, conn.to_node.id)]

            mutated_node.id = prev_id
            mutated_from_conn.innovation = from_innov
            mutated_to_conn.innovation = to_innov
        else:
            # Tracking the added node
            mutations[conn.innovation] = node_id
            # Tracking the added connections
            mutations[(conn.from_node.id, node_id)] = mutated_from_conn.innovation
            mutations[(node_id, conn.to_node.id)] = mutated_to_conn.innovation

            innov += 2
            node_id += 1

        self.add_connection(mutated_from_conn)
        self.add_connection(mutated_to_conn)
        self.add_node(mutated_node)

        return innov, node_id

    def _mutate_add_connection(self, node1: NEATNode, node2: NEATNode, mutations: dict, innov: int):
        from_node, to_node = (node1, node2) if node1.x < node2.x else (node2, node1)
        mutated_conn = Connection(from_node, to_node)

        # Make sure nodes aren't equal and there isn't already a connection between them
        # and the from node isn't an output node
        if node1.id == node2.id or \
                mutated_conn in self.connections or \
                Connection(to_node, from_node) in self.connections or \
                from_node.node_type == NodeType.OUTPUT:
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
        #
        # for mutagen in self.get_all_mutagens():
        #     mutagen.mutate()
        #
        # for node in self.nodes:
        #     node.mutate()
        #
        # for connection in self.connections:
        #     connection.mutate(self)

        return innov, node_id

    def to_phenotype(self, Phenotype):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """

        graph_node_map = {}
        root_node = None
        output_graph_node = None
        output_neat_node = None

        # initialises nodes and maps them to their genes
        for neat_node in self.nodes:
            if neat_node in graph_node_map:
                raise Exception("duplicate node in genome.nodes")

            graph_node_map[neat_node.id] = Phenotype(neat_node, self)

            if neat_node.is_input_node():
                if not (root_node is None):
                    raise Exception("multiple neat nodes labelled input node")
                root_node = graph_node_map[neat_node.id]
            if neat_node.is_output_node():
                if not (output_graph_node is None):
                    raise Exception("multiple neat nodes labelled output node")
                output_graph_node = graph_node_map[neat_node.id]
                output_neat_node = neat_node

        connected_input_node = False
        # connects the blueprint nodes as indicated by the genome
        connection_ids = set()
        for connection in self.connections:
            if not connection.enabled():
                # print("found disabled connection")
                continue
            if connection.from_node.id == connection.to_node.id or connection.to_node == connection.from_node:
                # raise Exception("connection from and to the same node", connection.from_node)
                continue

            conn_id = (connection.from_node.id, connection.to_node.id)
            if conn_id in connection_ids:
                # raise Exception("already connected neat node from",connection.from_node.id,"to",connection.to_node.id)
                continue

            connection_ids.add(conn_id)

            parent = graph_node_map[connection.from_node.id]
            child = graph_node_map[connection.to_node.id]

            parent.add_child(child)
            if parent == root_node:
                connected_input_node = True

        if not connected_input_node:
            print(self)
            raise Exception("no connections from input node", root_node)

        if not graph_node_map[output_neat_node.id].is_output_node():
            print(self)
            raise Exception("neat node marked as output has a graph node which is not an output node")

        sampled_trailing_node = root_node.get_output_node()
        while not sampled_trailing_node == output_graph_node:
            print("sampled a false output node:", sampled_trailing_node, "real output node:", output_graph_node)
            sampled_trailing_node.severe_node()
            sampled_trailing_node = root_node.get_output_node()
            if sampled_trailing_node == root_node:
                print(self)
                raise Exception("root node is output node - num children:", len(root_node.children))

        output_reaching_nodes = root_node.get_all_nodes_via_bottom_up(set())
        input_reaching_nodes = output_graph_node.get_all_nodes_via_top_down(set())

        fully_connected_nodes = output_reaching_nodes.intersection(input_reaching_nodes)

        if not output_graph_node in fully_connected_nodes:
            raise Exception("output node not in fully connected nodes")

        for neat_node in self.nodes:
            graph_node = graph_node_map[neat_node.id]
            if graph_node in fully_connected_nodes:
                continue
            if neat_node.node_type == NodeType.OUTPUT:
                raise Exception("severing the neat output node, is_graph_output_node:", graph_node.is_output_node())
            graph_node.severe_node()

        root_node.get_traversal_ids("_")
        return root_node

    def get_all_mutagens(self):
        return []

    def validate(self):
        node_dict = self.get_node_dict()
        for node in self.nodes:
            if node.node_type == NodeType.INPUT:
                input_node = node

            if node.node_type == NodeType.OUTPUT:
                output_node = node

        return self.from_input_to_output(input_node, output_node, node_dict, set())

    def get_node_dict(self):
        nodes_dict = {}  # dictionary maps node from node to all connected nodes
        for conn in self.connections:
            if not conn.enabled():
                continue

            if conn.from_node in nodes_dict:
                nodes_dict[conn.from_node].append(conn.to_node)
            else:
                nodes_dict[conn.from_node] = [conn.to_node]

        return nodes_dict

    def from_input_to_output(self, curr_node, output_node, nodes_dict, visited_nodes):
        if curr_node == output_node:
            return True

        if curr_node in visited_nodes:
            return False

        visited_nodes.add(curr_node)
        if curr_node not in nodes_dict:
            return False

        for node in nodes_dict[curr_node]:
            if self.from_input_to_output(node, output_node, nodes_dict, visited_nodes):
                return True

        return False

    def fix_height(self):
        node_dict = self.get_node_dict()

        for node in self.nodes:
            if node.node_type == NodeType.INPUT:
                input_node = node

        self._fix_height(input_node, node_dict)

    def _fix_height(self, curr_node, nodes_dict):
        if curr_node not in nodes_dict:
            return

        for node in nodes_dict[curr_node]:
            node.x = max(node.x, curr_node.x + 1)
            self._fix_height(node, nodes_dict)

    def __repr__(self):
        conns = ''
        for conn in self.connections:
            conns += str(conn) + '\n'

        return 'Connections: ' + conns[:-1] + '\nNodes' + str(self.nodes)


inp = NEATNode(0, 0, node_type=NodeType.INPUT)
outp = NEATNode(1, 1, node_type=NodeType.OUTPUT)
btw = NEATNode(2, 0, node_type=NodeType.HIDDEN)
g = Genome(
    [Connection(inp, outp, enabled=True, innovation=0),
     Connection(inp, btw, enabled=True, innovation=1),
     Connection(btw, outp, enabled=False, innovation=2)],
    [inp, outp, btw])

print(g.validate())
