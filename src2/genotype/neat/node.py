from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING, Set, Tuple

import random
from enum import Enum
from typing import Union

from src2.configuration import config
from src2.genotype.mutagen.option import Option
from src2.genotype.neat.gene import Gene

if TYPE_CHECKING:
    from src2.genotype.cdn.nodes.blueprint_node import BlueprintNode
    from src2.genotype.cdn.nodes.da_node import DANode
    from src2.genotype.cdn.nodes.module_node import ModuleNode


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Node(Gene):
    """General neat node"""

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

    def interpolate(self, other: Union[ModuleNode, BlueprintNode, DANode]):
        raise NotImplemented()
