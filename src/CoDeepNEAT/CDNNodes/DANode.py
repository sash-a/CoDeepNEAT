import random

from Config import Config
from NEAT.Gene import NodeGene, NodeType
from NEAT.Mutagen import Mutagen, ValueType


class DANode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN):
        """initialises da mutagens, and sets initial values"""
        super().__init__(id, node_type)

        da_submutagens = {

            "Rotate": {
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-30, start_range=-180,
                              end_range=-1, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=30, start_range=0,
                              end_range=180, mutation_chance=0.2)},

            "Translate_Pixels": {
                "x_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-4, start_range=-15,
                                end_range=-1, mutation_chance=0.2),
                "x_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=0,
                                end_range=15, mutation_chance=0.2),
                "y_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-4, start_range=-15,
                                end_range=-1, mutation_chance=0.2),
                "y_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=0,
                                end_range=15, mutation_chance=0.2)},

            "Scale": {
                "x_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.75, start_range=0.25,
                                end_range=0.99, mutation_chance=0.3),
                "x_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.25, start_range=1.0,
                                end_range=2.0, mutation_chance=0.3),
                "y_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.75, start_range=0.25,
                                end_range=0.99, mutation_chance=0.3),
                "y_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.25, start_range=1.0,
                                end_range=2.0, mutation_chance=0.3)},

            "Pad_Pixels": {
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=1, start_range=0,
                              end_range=3, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=4,
                              end_range=6, mutation_chance=0.2),
                "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.2)},

            "Crop_Pixels": {
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=1, start_range=0,
                              end_range=3, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=4,
                              end_range=6, mutation_chance=0.2),
                "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.2)},

            "Custom_Canny_Edges": {
                "min_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=100, start_range=50,
                                   end_range=149, mutation_chance=0.2),
                "max_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=200, start_range=150,
                                   end_range=250, mutation_chance=0.2)},

            "Additive_Gaussian_Noise": {
                "lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.05, start_range=0.0,
                              end_range=0.09, mutation_chance=0.3),
                "hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.125, start_range=0.10,
                              end_range=0.30, mutation_chance=0.3),
                "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                   end_range=0.8, mutation_chance=0.3)
            },

            "Coarse_Dropout": {
                "d_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.03, start_range=0.0,
                                end_range=0.09, mutation_chance=0.3),
                "d_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.15, start_range=0.1,
                                end_range=0.3, mutation_chance=0.3),
                "s_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.025, start_range=0.0,
                                end_range=0.09, mutation_chance=0.3),
                "s_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.3, start_range=0.1,
                                end_range=1.0, mutation_chance=0.3),
                "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                   end_range=0.8, mutation_chance=0.3)
            },

        }

        if Config.colour_augmentations:

            da_submutagens["HSV"] = {
                "channel": Mutagen(0, 1, 2, discreet_value=0, mutation_chance=0.1),
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=20, start_range=0,
                              end_range=29, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=50, start_range=30,
                              end_range=60, mutation_chance=0.2)
            }
            da_submutagens["Grayscale"] = {
                "alpha_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.35, start_range=0.0,
                                    end_range=0.49, mutation_chance=0.3),
                "alpha_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.75, start_range=0.5,
                                    end_range=1.0, mutation_chance=0.3)}

            self.da = Mutagen("Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                              "Grayscale", "Custom_Canny_Edges", "Additive_Gaussian_Noise", "Coarse_Dropout",
                              "HSV", "No_Operation", name="da type", sub_mutagens=da_submutagens,
                              discreet_value=random.choice(list(da_submutagens.keys())), mutation_chance=0.25)
        else:

            self.da = Mutagen("Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                              "Custom_Canny_Edges", "Additive_Gaussian_Noise", "Coarse_Dropout",
                              "No_Operation", name="da type", sub_mutagens=da_submutagens,
                              discreet_value=random.choice(list(da_submutagens.keys())), mutation_chance=0.25)

        self.enabled = Mutagen(True, False, discreet_value=True, name="da enabled")

    def get_all_mutagens(self):
        return [self.da, self.enabled]

    def get_node_name(self):
        return repr(self.da()) + "\n" + self.get_node_parameters()

    def get_node_parameters(self):
        """used for plotting da genomes"""
        parameters = []
        if self.da.get_sub_values() is not None:
            for key, value in self.da.get_sub_values().items():

                if value is None:
                    raise Exception("none value in mutagen")

                v = repr(value())

                parameters.append((key, v))

        return repr(parameters)