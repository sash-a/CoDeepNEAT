from enum import Enum


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Node:
    def __init__(self, id, node_type=NodeType.HIDDEN):
        self.id = id
        self.node_type = node_type

    def __eq__(self, other):
        return other.id == self.id
