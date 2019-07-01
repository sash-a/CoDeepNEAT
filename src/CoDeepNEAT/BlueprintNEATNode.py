from src.NEAT.NEATNode import NEATNode, NodeType
from src.NEAT.Mutagen import Mutagen
from src.NEAT.Mutagen import ValueType


class BlueprintNEATNode(NEATNode):

    def __init__(self, id, x, node_type=NodeType.HIDDEN):
        super(BlueprintNEATNode, self).__init__(id, x, node_type=node_type)
        self.species_number = Mutagen(value_type=ValueType.WHOLE_NUMBERS,current_value=0, start_range=0,
                                    end_range=100)


    def get_all_mutagens(self):
        return [self.species_number]
