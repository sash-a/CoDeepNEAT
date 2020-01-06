import random

from src2.configuration import config
from src2.phenotype.augmentations.da_submutagens import get_da_submutagens
from src2.genotype.mutagen.option import Option
from src2.genotype.neat.node import Node, NodeType
from src2.phenotype.augmentations.da_definitions import Augmentations


class DANode(Node):

    def __init__(self, id: int, type: NodeType):
        super().__init__(id, type)

        submutagens = get_da_submutagens()
        # Separated Photometric (Colour) augmentations from geometric (non-colour) ones
        self.da: Option
        if config.use_colour_augmentations:
            self.da: Option = Option("DA Type", "Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels",
                                     "Crop_Pixels", "Grayscale", "Coarse_Dropout", "HSV", "No_Operation",
                                     current_value=random.choice(list(submutagens.keys())),
                                     submutagens=submutagens, mutation_chance=0.25)
        else:
            self.da: Option = Option("DA Type", "Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels",
                                     "Crop_Pixels", "Coarse_Dropout", "No_Operation",
                                     current_value=random.choice(list(submutagens.keys())),
                                     submutagens=submutagens, mutation_chance=0.25)

    enabled = Option("enabled", True, False, current_value=True)

    def get_all_mutagens(self):
        return [self.da, self.enabled]

    def to_phenotype(self):
        kwargs = {k: mutagen.value for k, mutagen in self.da.submutagens[self.da.value].items()}
        print(self.da.value)
        print(kwargs)
        return Augmentations[self.da.value](**kwargs)

    # def get_node_name(self):
    #     return repr(self.da) + "\n" + self.get_node_parameters()

    # def get_node_parameters(self):
    #     """used for plotting DA genomes"""
    #     parameters = []
    #     if self.da.get_sub_values() is not None:
    #         for key, value in self.da.get_sub_values().items():
    #
    #             if value is None:
    #                 raise Exception("none value in mutagen")
    #
    #             v = repr(value())
    #
    #             parameters.append((key, v))
    #
    #     return repr(parameters)
