import random

from torch import nn

from src2.Configuration import config
from src2.Genotype.Mutagen.ContinuousVariable import ContinuousVariable
from src2.Genotype.Mutagen.IntegerVariable import IntegerVariable
from src2.Genotype.Mutagen.Option import Option
from src2.Genotype.NEAT.Node import Node, NodeType


def get_conv_parameter_mutagens():
    return {
        "conv_window_size": Option("conv_window_size", 3, 5, 7, current_value=random.choice([3, 5, 7]),
                                   mutation_chance=0.13),

        "conv_stride": IntegerVariable("conv_stride", current_value=1, start_range=1, end_range=5, mutation_chance=0.1),

        "reduction": Option("reduction", None, nn.MaxPool2d,
                            current_value=nn.MaxPool2d if random.random() < config.module_node_max_pool_chance else None,
                            submutagens=
                            {
                                nn.MaxPool2d: {"pool_size": IntegerVariable("pool_size", current_value=2, start_range=2,
                                                                            end_range=5, mutation_chance=0.1)}
                            }
                            , mutation_chance=0.15),

        "regularisation": Option("regularisation", None, nn.BatchNorm2d,
                                 current_value=nn.BatchNorm2d if random.random() < config.module_node_batchnorm_chance else None,
                                 mutation_chance=0.15),

        "dropout": Option("dropout", None, nn.Dropout2d,
                          current_value=nn.Dropout2d if random.random() < config.module_node_dropout_chance else None,
                          submutagens=
                          {
                              nn.Dropout2d: {
                                  "dropout_factor": ContinuousVariable("dropout_factor", current_value=0.1,
                                                                       start_range=0, end_range=0.75,
                                                                       mutation_chance=0.15)
                              }
                          },
                          mutation_chance=0.08),

        "out_features": IntegerVariable("out_features", current_value=random.normalvariate(50, 20), start_range=1,
                                        end_range=100, mutation_chance=0.22)
    }


def get_linear_parameter_mutagens():
    return {

    }


def get_layer_submutagens():
    return {
        nn.Conv2d: get_conv_parameter_mutagens(),
        nn.Linear: get_linear_parameter_mutagens()
    }


class ModuleNode(Node):

    def __int__(self, id: int, type: NodeType):
        super().__init__(id, type)

        layer_type = None if random.random() > config.module_node_deep_layer_chance else (
            nn.Conv2d if random.random() < config.module_node_conv_layer_chance else nn.Linear)
        self.layer_type = Option("layer_type", None, nn.Conv2d, nn.Linear, current_value=layer_type,
                                 submutagens=get_layer_submutagens())
