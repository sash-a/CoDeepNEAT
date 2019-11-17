import random
from enum import Enum

from src2.Genotype.NEAT.Gene import Gene
from src2.Genotype.Mutagen.Option import Option


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Node(Gene):
    """General NEAT node"""

    def __init__(self, id, type: NodeType = NodeType.HIDDEN):
        super().__init__(id)
        self.node_type: NodeType = type
        self.lossy_aggregation = Option('lossy', True, False, current_value=random.choice([True, False]))
        self.try_conv_aggregation = Option('conv_aggregation', True, False, current_value=random.choice([True, False]))

    def is_output_node(self):
        return self.node_type == NodeType.OUTPUT

    def is_input_node(self):
        return self.node_type == NodeType.INPUT

    def get_all_mutagens(self):
        return []

    def convert_node(self, **kwargs):
        raise NotImplemented()
