from enum import Enum

from src2.Genotype.NEAT.Gene import Gene


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Node(Gene):
    """General NEAT node"""

    def __init__(self, id, type: NodeType = NodeType.HIDDEN):
        super().__init__(id)

        self.height = -1
        self.node_type = type

    def is_output_node(self):
        return self.node_type == NodeType.OUTPUT

    def is_input_node(self):
        return self.node_type == NodeType.INPUT

    def get_all_mutagens(self):
        return []
