from src.NEAT.NEATNode import NEATNode
from src.NEAT.NEATNode import NodeType


class BlueprintNEATNode(NEATNode):

    def __init__(self, id, x, node_type=NodeType.HIDDEN):
        super(BlueprintNEATNode, self).__init__(id, x, node_type=node_type)
        self.species_number = 0

    pass
