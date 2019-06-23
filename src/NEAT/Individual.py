from enum import Enum


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


class NeatNode:
    def __init__(self, id, node_type=NodeType.HIDDEN):
        self.id = id
        self.node_type = node_type


class NeatIndividual:
    def __init__(self, connections):
        self.connections = connections

    def crossover(self, other_individual):
        pass

    def mutate(self):
        pass


def main():
    nodes = [NeatNode(0, NodeType.INPUT), NeatNode(1, NodeType.HIDDEN), NeatNode(2, NodeType.OUTPUT)]


if __name__ == '__main__':
    main()
