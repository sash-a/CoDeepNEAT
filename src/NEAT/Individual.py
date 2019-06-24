from enum import Enum

compat_thresh = 1


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class NeatConnection:
    def __init__(self, in_node, out_node, connection_type=None, enabled=True, innovation=0):
        self.in_node = in_node
        self.out_node = out_node
        self.connection_type = connection_type
        self.enabled = enabled
        self.innovation = innovation

    def __repr__(self):
        return 'innovation: ' + str(self.innovation)


class NeatNode:
    def __init__(self, id, node_type=NodeType.HIDDEN):
        self.id = id
        self.node_type = node_type


class NeatIndividual:
    def __init__(self, connections):
        self.fitness = 0
        self.adjusted_fitness = 0

        self.connections = []
        for connection in connections:
            self.add_connection(connection)

    def add_connection(self, new_connection):
        """
        Adds a new connection maintaining the order of the connections list

        :param new_connection: The new connection to add
        :return: The position the new connection was added at
        """
        pos = 0
        for i, connection in enumerate(self.connections, 0):
            if new_connection.innovation < connection.innovation:
                break
            pos += 1

        self.connections.insert(pos, new_connection)
        return pos

    def get_max_innov(self):
        return self.connections[-1].innovation

    # This is not as efficient but it is cleaner than a single pass
    def get_disjoint_excess(self, other_individual):
        """
        Finds all of self's disjoint and excess genes

        :param other_individual: The individual to compare to
        :return: tuple containing list of disjoint and a list of excess genes
        """
        disjoint = []
        excess = []

        other_innovs = set([connection.innovation for connection in other_individual.connections])

        for connection in self.connections:
            if connection.innovation not in other_innovs:
                if connection.innovation > other_individual.get_max_innov():
                    excess.append(connection)
                else:
                    disjoint.append(connection)

        return disjoint, excess

    # disjoint + excess are inherited from the most fit parent
    def crossover(self, individual):
        pass

    def mutate(self):
        pass


class NeatSpecies:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]

    def add_member(self, new_member):
        if self.is_compatible(new_member):
            self.members.append(new_member)

    def __add__(self, other):
        self.add_member(other)

    def is_compatible(self, individual):
        n = max(len(self.representative.connections), len(individual.connections))
        self_d, self_e = self.representative.get_disjoint_excess(individual)
        other_d, other_e = individual.get_disjoint_excess(self.representative)

        d = len(self_d) + len(other_d)
        e = len(self_e) + len(other_e)

        compatibility = (d + e) / n
        return compatibility < compat_thresh


def speciate(individuals):
    # place g in first compatible species otherwise create new one with g as representative
    pass


def main():
    nodes = [NeatNode(0, NodeType.INPUT), NeatNode(1, NodeType.HIDDEN), NeatNode(2, NodeType.OUTPUT)]
    connections = \
        [
            NeatConnection(nodes[0], nodes[1], innovation=0),
            NeatConnection(nodes[1], nodes[2], innovation=1),
        ]

    indv1 = NeatIndividual(connections)
    indv2 = NeatIndividual([NeatConnection(nodes[1], nodes[2], innovation=1)])

    spc = NeatSpecies(indv1)
    print(spc.is_compatible(indv2))


if __name__ == '__main__':
    main()
