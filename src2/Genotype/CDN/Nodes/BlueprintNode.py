from src2.Genotype.Mutagen.IntegerVariable import IntegerVariable
from src2.Genotype.NEAT.Node import Node, NodeType


class BlueprintNode(Node):

    def __init__(self, id: int, type: NodeType):
        super().__init__(id, type)

        self.linked_module_id: int = -1
        self.module_repeat_count = IntegerVariable("module_repeat_count", start_range=1, current_value=1, end_range=4)
