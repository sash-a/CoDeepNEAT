from enum import Enum


class NodeType(Enum):  # TODO assign node type
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class NEATNode:

    def __init__(self, id, x, node_type=NodeType.HIDDEN):
        self.id = id
        self.x = x
        self.node_type = node_type

    def midpoint(self, other):
        return abs(self.x + other.x) / 2

    def __eq__(self, other):
        return other.id == self.id  # and self.x == other.x

    def __hash__(self):
        return self.id

    def __repr__(self):
        return str(self.id)

    def is_input_node(self):
        return self.node_type == NodeType.INPUT

    def is_output_node(self):
        return self.node_type == NodeType.OUTPUT

    def get_all_mutagens(self):
        raise Exception('Get all mutagens called in base class')

    def mutate(self):
        for mutagen in self.get_all_mutagens():
            mutagen.mutate()
