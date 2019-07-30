import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Config import Config, NeatProperties as Props
from src.NEAT.Gene import NodeGene, NodeType
from src.NEAT.Mutagen import Mutagen
from src.NEAT.Mutagen import ValueType

use_convs = True
use_linears = True


class ModulenNEATNode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN, activation=F.relu, layer_type=nn.Conv2d,
                 conv_window_size=7, conv_stride=1, max_pool_size=2):
        super(ModulenNEATNode, self).__init__(id, node_type)

        batch_norm_chance = 0.65  # chance that a new node will start with batch norm
        use_batch_norm = random.random() < batch_norm_chance

        dropout_chance = 0.2  # chance that a new node will start with drop out
        use_dropout = random.random() < dropout_chance

        max_pool_chance = 0.3  # chance that a new node will start with drop out
        use_max_pool = random.random() < dropout_chance

        self.activation = Mutagen(F.relu, F.leaky_relu, torch.sigmoid, F.relu6,
                                  discreet_value=activation, name="activation function",
                                  mutation_chance=0.15)  # TODO try add in Selu, Elu

        conv_out_features = 25 + random.randint(0, 25)
        linear_out_features = 100 + random.randint(0, 100)

        linear_submutagens = \
            {
                "regularisation": Mutagen(None, nn.BatchNorm1d,
                                          discreet_value=nn.BatchNorm1d if use_batch_norm else None,
                                          mutation_chance=0.15),

                "dropout": Mutagen(None, nn.Dropout, discreet_value=nn.Dropout if use_dropout else None, sub_mutagens=
                {
                    nn.Dropout: {
                        "dropout_factor": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.15, start_range=0,
                                                  end_range=0.75)}
                }, mutation_chance=0.08),

                "out_features": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=linear_out_features,
                                        start_range=10,
                                        end_range=1024, name="num out features", mutation_chance=0.22,
                                        distance_weighting=Props.LAYER_SIZE_COEFFICIENT)
            }

        conv_submutagens = {
            "conv_window_size": Mutagen(3, 5, 7, discreet_value=conv_window_size, mutation_chance=0.13),

            "conv_stride": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=conv_stride, start_range=1,
                                   end_range=5),

            "reduction": Mutagen(None, nn.MaxPool2d, discreet_value=nn.MaxPool2d if use_max_pool else None,
                                 sub_mutagens=
                                 {
                                     nn.MaxPool2d: {"pool_size": Mutagen(
                                         value_type=ValueType.WHOLE_NUMBERS, current_value=max_pool_size, start_range=2,
                                         end_range=5)}
                                 }, mutation_chance=0.15),

            "regularisation": Mutagen(None, nn.BatchNorm2d, discreet_value=nn.BatchNorm2d if use_batch_norm else None,
                                      mutation_chance=0.15),

            "dropout": Mutagen(None, nn.Dropout2d, discreet_value=nn.Dropout2d if use_dropout else None, sub_mutagens=
            {
                nn.Dropout2d: {
                    "dropout_factor": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.1,
                                              start_range=0, end_range=0.75)}
            }, mutation_chance=0.08),

            "out_features": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=conv_out_features, start_range=1,
                                    end_range=100, name="num out features", mutation_chance=0.22,
                                    distance_weighting=Props.LAYER_SIZE_COEFFICIENT)
        }

        if use_linears and not use_convs:
            self.layer_type = Mutagen(nn.Linear, discreet_value=nn.Linear,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT,
                                      sub_mutagens={nn.Linear: linear_submutagens}
                                      )
        if use_convs and not use_linears:
            self.layer_type = Mutagen(nn.Conv2d, discreet_value=nn.Conv2d,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT,
                                      sub_mutagens={nn.Conv2d: conv_submutagens})
        if use_convs and use_linears:
            self.layer_type = Mutagen(nn.Conv2d, nn.Linear, discreet_value=layer_type,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT,
                                      sub_mutagens={
                                          nn.Conv2d: conv_submutagens,
                                          nn.Linear: linear_submutagens
                                      }, name="deep layer type", mutation_chance=0.08)

    def get_all_mutagens(self):
        return [self.activation, self.layer_type]


class BlueprintNEATNode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN):
        super(BlueprintNEATNode, self).__init__(id, node_type)

        self.species_number = Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=0, start_range=0,
                                      end_range=1, print_when_mutating=False, name="species number",
                                      mutation_chance=0.13)

    def get_all_mutagens(self):
        # raise Exception("getting species no mutagen from blueprint neat node")
        return [self.species_number]

    def set_species_upper_bound(self, num_species):
        self.species_number.end_range = num_species
        self.species_number.set_value(min(self.species_number(), num_species - 1))


class DANode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN):
        super().__init__(id, node_type)

        if Config.colour_augmentations:
            self.da = Mutagen("Flip_lr", "Flip_ud", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                              "Grayscale", "Custom_Canny_Edges", "Shear", "Additive_Gaussian_Noise",
                              "Coarse_Dropout", "HSV", "Contrast_Normalisation", "Increase_Channel", "Rotate_Channel",
                              "No_Operation", name="da type", sub_mutagens={

                    "Rotate": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-45, start_range=-180,
                                      end_range=0),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=+45, start_range=0,
                                      end_range=180)},

                    "Translate_Pixels": {
                        "x_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-10, start_range=-25,
                                        end_range=0),
                        "x_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=10, start_range=0,
                                        end_range=25),
                        "y_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-10, start_range=-25,
                                        end_range=0),
                        "y_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=10, start_range=0,
                                        end_range=25)},

                    "Scale": {
                        "x_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.25,
                                        end_range=1.0),
                        "x_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.5, start_range=1.0,
                                        end_range=2.0),
                        "y_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.25,
                                        end_range=1.0),
                        "y_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.5, start_range=1.0,
                                        end_range=2.0)},

                    "Pad_Pixels": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=2, start_range=0,
                                      end_range=5),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=7, start_range=5,
                                      end_range=10),
                        "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.25)},

                    "Crop_Pixels": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=2, start_range=0,
                                      end_range=5),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=7, start_range=5,
                                      end_range=10),
                        "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.25)},

                    "Grayscale": {
                        "alpha_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.0,
                                            end_range=0.5),
                        "alpha_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.0, start_range=0.5,
                                            end_range=1.0)},

                    "Custom_Canny_Edges": {
                        "min_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=100, start_range=50,
                                           end_range=150),
                        "max_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=200, start_range=150,
                                           end_range=250)},

                    "Shear": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-20, start_range=-40,
                                      end_range=0),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=20, start_range=0,
                                      end_range=40)},

                     "Additive_Gaussian_Noise": {
                        "lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.05, start_range=0.0,
                                      end_range=0.10),
                        "hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.20, start_range=0.10,
                                      end_range=0.40),
                        "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                           end_range=0.8)
                    },

                    "Coarse_Dropout": {
                        "d_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.05, start_range=0.0,
                                        end_range=0.1),
                        "d_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.2, start_range=0.1,
                                        end_range=0.3),
                        "s_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.025, start_range=0.0,
                                        end_range=0.1),
                        "s_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.1,
                                        end_range=1.0),
                        "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                           end_range=0.8)
                    },

                    "HSV": {
                        "channel": Mutagen(0, 1, 2, discreet_value=0, mutation_chance=0.20),
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=20, start_range=0, end_range=30),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=50, start_range=30, end_range=60)
                    },

                    "Contrast_Normalisation": {
                        "lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.0, end_range=1.0),
                        "hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.5, start_range=1.0, end_range=2.0),
                        "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.0, end_range=1.0)
                    },

                    "Increase_Channel": {
                        "channel": Mutagen(0, 1, 2, discreet_value=0, mutation_chance=0.20),
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=25, start_range=0, end_range=50),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=75, start_range=50, end_range=100)
                    },

                    "Rotate_Channel": {
                        "channel": Mutagen(0, 1, 2, discreet_value=0, mutation_chance=0.20),
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-45, start_range=-180,
                                      end_range=0),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=+45, start_range=0,
                                      end_range=180)
                    }

                },
                              discreet_value="Flip_lr")

            self.enabled = Mutagen(True, False, discreet_value=True, name="da enabled")

        else:

            self.da = Mutagen("Flip_lr", "Flip_ud", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                              "Custom_Canny_Edges", "Shear", "Additive_Gaussian_Noise", "Coarse_Dropout",
                              "No_Operation", name="da type", sub_mutagens={

                    "Rotate": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-45, start_range=-180,
                                      end_range=0),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=+45, start_range=0,
                                      end_range=180)},

                    "Translate_Pixels": {
                        "x_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-7, start_range=-15,
                                        end_range=0),
                        "x_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=7, start_range=0,
                                        end_range=15),
                        "y_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-7, start_range=-15,
                                        end_range=0),
                        "y_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=7, start_range=0,
                                        end_range=15)},

                    "Scale": {
                        "x_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.25,
                                        end_range=1.0),
                        "x_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.5, start_range=1.0,
                                        end_range=2.0),
                        "y_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.25,
                                        end_range=1.0),
                        "y_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.5, start_range=1.0,
                                        end_range=2.0)},

                    "Pad_Pixels": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=2, start_range=0,
                                      end_range=4),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=4,
                                      end_range=8),
                        "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.25)},

                    "Crop_Pixels": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=2, start_range=0,
                                      end_range=4),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=4,
                                      end_range=8),
                        "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.25)},

                    "Custom_Canny_Edges": {
                        "min_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=100, start_range=50,
                                           end_range=150),
                        "max_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=200, start_range=150,
                                           end_range=250)},

                    "Shear": {
                        "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-20, start_range=-40,
                                      end_range=0),
                        "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=20, start_range=0,
                                      end_range=40)},

                    "Additive_Gaussian_Noise": {
                        "lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.05, start_range=0.0,
                                      end_range=0.10),
                        "hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.20, start_range=0.10,
                                      end_range=0.40),
                        "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                           end_range=0.8)
                    },

                    "Coarse_Dropout": {
                        "d_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.05, start_range=0.0,
                                        end_range=0.1),
                        "d_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.2, start_range=0.1,
                                        end_range=0.3),
                        "s_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.025, start_range=0.0,
                                        end_range=0.1),
                        "s_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.1,
                                        end_range=1.0),
                        "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                           end_range=0.8)
                    },

                },
                              discreet_value="Flip_lr")

            self.enabled = Mutagen(True, False, discreet_value=True, name="da enabled")

    def get_all_mutagens(self):
        return [self.da, self.enabled]

    def get_node_name(self):
        return repr(self.da())
