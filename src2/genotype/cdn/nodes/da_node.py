import src2.phenotype.augmentations.evolved_augmentations as EDA
from src2.genotype.mutagen.option import Option
from src2.genotype.neat.node import Node, NodeType


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
