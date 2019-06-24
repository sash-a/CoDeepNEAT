import random
from src.NEAT.Connection import Connection
from src.NEAT.Node import Node
from src.NEAT.Mutation import NodeMutation, ConnectionMutation


class Genome:
    def __init__(self, connections, nodes):
        self.fitness = 0
        self.adjusted_fitness = 0

        self.nodes = nodes
        self.node_ids = set([node.id for node in nodes])

        self.innov_nums = set()
        self.connections = []
        for connection in connections:
            self.add_connection(connection)

    def add_connection(self, new_connection):
        """Adds a new connection maintaining the order of the connections list"""
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

    def add_node(self, node):
        self.node_ids.add(node.id)
        self.nodes.append(node)

    # TODO split into 2 methods
    # This is not as efficient but it is cleaner than a single pass
    def get_disjoint_excess(self, other):
        """Finds all of self's disjoint and excess genes when compared to other"""
        disjoint = []
        excess = []

        for connection in self.connections:
            if connection.innovation not in other.innov_nums:
                if connection.innovation > other.connections[-1].innovation:
                    excess.append(connection)
                else:
                    disjoint.append(connection)

        return disjoint, excess

    def adjust_fitness(self, fitness=None):
        if fitness is not None:
            self.fitness = fitness
        # TODO

    # TODO this should not belong to Genome
    def crossover(self, other_parent):
        # Choosing the fittest parent
        if other_parent.adjusted_fitness == self.adjusted_fitness:
            best_parent = self if len(self.connections) < other_parent.connections else other_parent
        else:
            best_parent = self if self.fitness > other_parent.fitness else other_parent

        # disjoint + excess are inherited from the most fit parent
        d, e = best_parent.get_disjoint_excess(self)  # TODO create copy
        child = Genome(d + e, list(set(self.nodes + other_parent.nodes)))

        for conn in self.connections:
            if conn.innovation in other_parent.innov_nums:
                other_conn = other_parent.get_connection(conn.innovation)  # TODO create copy
                child.add_connection(random.choice([conn, other_conn]))

        return child

    def mutate(self, curr_gen_mutations: set, node_chance=0.03, conn_chance=0.5):
        chance = random.randint(0, 1)

        # Add a new node
        if chance < node_chance:
            conn = random.choice(self.connections)
            conn.enabled = False
            # TODO innovation numbers and node ID
            mutated_node = Node(1)  # TODO global ID
            conn_from = Connection(conn.in_node, mutated_node)
            conn_to = Connection(mutated_node, conn.out_node)

            self.add_connection(conn_from)
            self.add_connection(conn_to)
            self.add_node(mutated_node)

            return NodeMutation(mutated_node.id, conn_from.innovation, conn_to.innovation)
        # Add a new connection
        elif chance > conn_chance:
            node1 = random.choice(self.nodes)
            node2 = random.choice(self.nodes)

            if node1 != node2:  # and there is not already a connection and connection does not create cycle
                # create connection (direction?)
                pass
