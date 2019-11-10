from src2.Genotype.NEAT.Node import Node, NodeType
import src2.Phenotype.Augmentations.EvolvedAugmentations as EDA
from src2.Genotype.Mutagen.Option import Option

class DANode(Node):

    def __init__(self, id: int, type: NodeType):
        super().__init__(id, type)

    enabled = Option("enabled", True, False, current_value=True)

    def get_all_mutagens(self):
        return [EDA.DA_Mutagens, self.enabled]

    def get_node_name(self):
        return repr(EDA.DA_Mutagens()) + "\n" + self.get_node_parameters()

    def get_node_parameters(self):
        """used for plotting DA genomes"""
        parameters = []
        if EDA.DA_Mutagens.get_sub_values() is not None:
            for key, value in EDA.DA_Mutagens.get_sub_values().items():

                if value is None:
                    raise Exception("none value in mutagen")

                v = repr(value())

                parameters.append((key, v))

        return repr(parameters)

    def pick_module(self):
        pass
