import random
from enum import Enum

from src2.Configuration import config
from src2.Genotype.Mutagen.Option import Option
from src2.Genotype.NEAT.Gene import Gene


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Node(Gene):
    """General NEAT node"""

    def __init__(self, id, type: NodeType = NodeType.HIDDEN):
        super().__init__(id)
        self.node_type: NodeType = type
        # TODO
        self.lossy_aggregation = Option('lossy', False, True,
                                        current_value=random.choices([False, True], weights=[1-config.lossy_chance,
                                                                                             config.lossy_chance])[0],
                                        mutation_chance=0.3 if config.mutate_lossy_values else 0)
        self.try_conv_aggregation = Option('conv_aggregation', False, True, current_value=random.choice([False, True]))

    def is_output_node(self):
        return self.node_type == NodeType.OUTPUT

    def is_input_node(self):
        return self.node_type == NodeType.INPUT

    def get_all_mutagens(self):
        return [self.lossy_aggregation, self.try_conv_aggregation]

    def convert_node(self, **kwargs):
        raise NotImplemented()
