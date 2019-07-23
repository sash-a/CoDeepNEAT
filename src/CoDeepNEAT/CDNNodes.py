from src.NEAT.Gene import NodeGene, NodeType
from src.NEAT.Gene import NodeGene, NodeType
from src.NEAT.Mutagen import Mutagen
from src.NEAT.Mutagen import ValueType
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

from src.DataAugmentation.AugmentationScheme import AugmentationScheme

use_convs = True
use_linears = True


class ModulenNEATNode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN,
                 out_features=25, activation=F.relu, layer_type=nn.Conv2d,
                 conv_window_size=7, conv_stride=1, max_pool_size=2):
        super(ModulenNEATNode, self).__init__(id, node_type)

        batch_norm_chance = 0.65
        use_batch_norm = random.random() < batch_norm_chance

        dropout_chance = 0.3
        use_dropout = random.random() < dropout_chance

        self.activation = Mutagen(F.relu, F.leaky_relu, torch.sigmoid, F.relu6,
                                  discreet_value=activation, name="activation function")  # TODO try add in Selu, Elu

        linear_submutagens = {
            "regularisation": Mutagen(None, nn.BatchNorm1d, discreet_value=nn.BatchNorm1d if use_batch_norm else None),
            "dropout": Mutagen(None, nn.Dropout, sub_mutagens={nn.Dropout: {
                "dropout_factor": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.15,
                                          start_range=0, end_range=0.75)}},
                               discreet_value=nn.Dropout if use_dropout else None),
            "out_features": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=100,
                                    start_range=10,
                                    end_range=1024, name="num out features")}

        conv_submutagens = {"conv_window_size": Mutagen(3, 5, 7, discreet_value=conv_window_size),
                            "conv_stride": Mutagen(value_type=ValueType.WHOLE_NUMBERS,
                                                   current_value=conv_stride, start_range=1,
                                                   end_range=5),
                            "reduction": Mutagen(None, nn.MaxPool2d, discreet_value=None,
                                                 sub_mutagens={nn.MaxPool2d: {
                                                     "pool_size": Mutagen(
                                                         value_type=ValueType.WHOLE_NUMBERS,
                                                         current_value=max_pool_size, start_range=2,
                                                         end_range=5)}}),
                            "regularisation": Mutagen(None, nn.BatchNorm2d,
                                                      discreet_value=nn.BatchNorm2d if use_batch_norm else None),
                            "dropout": Mutagen(None, nn.Dropout2d, sub_mutagens={
                                nn.Dropout2d: {
                                    "dropout_factor": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.1,
                                                              start_range=0, end_range=0.75)}},
                                               discreet_value=nn.Dropout2d if use_dropout else None),
                            "out_features": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=out_features,
                                                    start_range=1,
                                                    end_range=100, name="num out features")
                            }

        if use_linears and not use_convs:
            self.layer_type = Mutagen(nn.Linear, discreet_value=nn.Linear, sub_mutagens={
                nn.Linear: linear_submutagens
            })
        if use_convs and not use_linears:
            self.layer_type = Mutagen(nn.Conv2d, discreet_value=nn.Conv2d,
                                      sub_mutagens={nn.Conv2d: conv_submutagens})
        if use_convs and use_linears:
            self.layer_type = Mutagen(nn.Conv2d, nn.Linear, discreet_value=layer_type,
                                      sub_mutagens={
                                          nn.Conv2d: conv_submutagens,
                                          nn.Linear: linear_submutagens
                                      }, name="deep layer type")

    def get_all_mutagens(self):
        return [self.activation, self.layer_type]


class BlueprintNEATNode(NodeGene):

    def __init__(self, id, node_type=NodeType.HIDDEN):
        super(BlueprintNEATNode, self).__init__(id, node_type)

        self.species_number = Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=0, start_range=0,
                                      end_range=1, print_when_mutating=False, name="species number")



    def get_all_mutagens(self):
        # raise Exception("getting species no mutagen from blueprint neat node")
        return [self.species_number]

    def set_species_upper_bound(self, num_species):
        self.species_number.end_range = num_species
        self.species_number.set_value(min(self.species_number(), num_species - 1))


class DANode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN):
        super().__init__(id, node_type)
        # self.da = Mutagen(*list(AugmentationScheme.Augmentations.keys()), discreet_value='No_Operation')
        self.da = Mutagen("Flip_lr", "Flip_ud", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                          "Grayscale", "Custom_Canny_Edges", "Shear", "Additive_Gaussian_Noise",
                          "Coarse_Dropout", "No_Operation", name="da type", sub_mutagens={

                "Rotate": {
                    "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-45, start_range=-180, end_range=0),
                    "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=+45, start_range=0, end_range=180)},

                "Translate_Pixels": {
                    "x_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-20, start_range=-50,
                                    end_range=0),
                    "x_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=20, start_range=0, end_range=50),
                    "y_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-20, start_range=-50,
                                    end_range=0),
                    "y_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=20, start_range=0, end_range=50)},

                "Scale": {
                    "x_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.0, end_range=1.0),
                    "x_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.5, start_range=1.0, end_range=2.0),
                    "y_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.0, end_range=1.0),
                    "y_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.5, start_range=1.0,
                                    end_range=2.0)},

                "Pad_Pixels": {
                    "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=10, start_range=0, end_range=25),
                    "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=30, start_range=25, end_range=50),
                    "s_i": Mutagen(True, False, discreet_value=False)},

                "Crop_Pixels": {
                    "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=10, start_range=0, end_range=25),
                    "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=30, start_range=25, end_range=50),
                    "s_i": Mutagen(True, False, discreet_value=False)},

                "Grayscale": {"alpha_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.0,
                                                  end_range=0.5),
                              "alpha_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.0, start_range=0.5,
                                                  end_range=1.0)},

                "Custom_Canny_Edges": {
                    "min_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=100, start_range=0,
                                       end_range=150),
                    "max_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=200, start_range=150,
                                       end_range=250)},

                "Shear": {
                    "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-15, start_range=-30, end_range=0),
                    "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=15, start_range=0, end_range=30)},

                "Additive_Gaussian_Noise": {
                    "lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.0, start_range=0.0, end_range=0.5),
                    "hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.5, end_range=1.0)},

                "Coarse_Dropout": {
                    "d_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.0, end_range=0.1),
                    "d_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.1, start_range=0.1, end_range=0.3),
                    "s_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.025, start_range=0.0,
                                    end_range=0.25),
                    "s_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.5, start_range=0.25,
                                    end_range=0.75)}

            },
                          discreet_value="Flip_lr")

        self.enabled = Mutagen(True, False, discreet_value=True, name="da enabled")

    def get_all_mutagens(self):
        return [self.da, self.enabled]
