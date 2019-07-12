from src.NEAT.Gene import NodeGene, NodeType
from NEAT.Mutagen import Mutagen
from NEAT.Mutagen import ValueType


class BlueprintNEATNode(NodeGene):

    def __init__(self, id, node_type=NodeType.HIDDEN):
        super(BlueprintNEATNode, self).__init__(id, node_type)

        # print("module pop:",Generation.module_population)
        self.species_number = Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=0, start_range=0,
                                      end_range=1, print_when_mutating=False)

    def get_all_mutagens(self):
        return [self.species_number]

    def set_species_upper_bound(self, num_species):
        self.species_number._end_range = num_species
