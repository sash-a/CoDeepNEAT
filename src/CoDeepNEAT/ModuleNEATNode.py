from src.NEAT.NEATNode import NEATNode
from enum import Enum

class ModulenNEATNode(NEATNode):

    class NodeType(Enum):
        INPUT = 0
        HIDDEN = 1
        OUTPUT = 2

    pass