import random

import torch
from torch import nn as nn
from torch.nn import functional as F

from src.Config import NeatProperties as Props, Config
from src.NEAT.Gene import NodeGene, NodeType
from src.NEAT.Mutagen import Mutagen, ValueType

use_convs = True
use_linears = True


class ModuleNEATNode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN, activation=F.relu, layer_type=nn.Conv2d,
                 conv_window_size=3, conv_stride=1, max_pool_size=2):
        """initialises all of the nodes mutagens, and assigns initial values"""

        super(ModuleNEATNode, self).__init__(id, node_type)

        batch_norm_chance = 0.65  # chance that a new node will start with batch norm
        use_batch_norm = random.random() < batch_norm_chance

        dropout_chance = 0.2  # chance that a new node will start with drop out
        use_dropout = random.random() < dropout_chance

        max_pool_chance = 0.3  # chance that a new node will start with drop out
        use_max_pool = random.random() < max_pool_chance

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
                                        distance_weighting=Props.LAYER_SIZE_COEFFICIENT if Config.allow_attribute_distance else 0)
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
                                    distance_weighting=Props.LAYER_SIZE_COEFFICIENT if Config.allow_attribute_distance else 0)
        }

        if use_linears and not use_convs:
            self.layer_type = Mutagen(nn.Linear, discreet_value=nn.Linear,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT if Config.allow_attribute_distance else 0,
                                      sub_mutagens={nn.Linear: linear_submutagens}
                                      )
        if use_convs and not use_linears:
            self.layer_type = Mutagen(nn.Conv2d, discreet_value=nn.Conv2d,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT if Config.allow_attribute_distance else 0,
                                      sub_mutagens={nn.Conv2d: conv_submutagens})
        if use_convs and use_linears:
            self.layer_type = Mutagen(nn.Conv2d, nn.Linear, discreet_value=layer_type,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT if Config.allow_attribute_distance else 0,
                                      sub_mutagens={
                                          nn.Conv2d: conv_submutagens,
                                          nn.Linear: linear_submutagens
                                      }, name="deep layer type", mutation_chance=0.08)

    def get_all_mutagens(self):
        return [self.activation, self.layer_type]

    def __repr__(self):
        return str(self.node_type)

    def get_node_name(self):
        return repr(self.layer_type()) + "\n" + "features: " + repr(self.layer_type.get_sub_value("out_features"))

    def get_complexity(self):
        """approximates the size of this layer in terms of trainable parameters"""
        if self.layer_type() == nn.Conv2d:
            return pow(self.layer_type.get_sub_value("conv_window_size"), 2) * self.layer_type.get_sub_value(
                "out_features")
        elif self.layer_type() == nn.Linear:
            return self.layer_type.get_sub_value("out_features")
        else:
            raise Exception()
