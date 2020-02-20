from __future__ import annotations

import random
from typing import Tuple, TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from configuration import config
from src.genotype.mutagen.continuous_variable import ContinuousVariable
from src.genotype.mutagen.integer_variable import IntegerVariable
from src.genotype.mutagen.option import Option
from src.genotype.neat.node import Node, NodeType
from src.phenotype.neural_network.layers.custom_layer_types.depthwise_separable_conv import DepthwiseSeparableConv
from src.phenotype.neural_network.layers.layer import Layer

if TYPE_CHECKING:
    pass


class ModuleNode(Node):
    def __init__(self, id: int, type: NodeType):
        super().__init__(id, type)
        layer_type = None if random.random() > config.module_node_deep_layer_chance else (
            nn.Conv2d if random.random() < config.module_node_conv_layer_chance else nn.Linear)
        layer_types = [None, nn.Conv2d, nn.Linear]

        if config.use_depthwise_separable_convs:
            layer_types.append(DepthwiseSeparableConv)

        self.layer_type: Option = Option("layer_type", *layer_types, current_value=layer_type,
                                         submutagens=get_new_layer_submutagens(),
                                         mutation_chance=config.module_node_layer_type_change_chance)

        self.activation: Option = Option("activation", F.relu, F.leaky_relu, torch.sigmoid, F.relu6,
                                         current_value=F.leaky_relu, mutation_chance=0.15)  # TODO try add in Selu, Elu

        self.layer_repeats = IntegerVariable("layer_repeats", start_range=1, current_value=1,
                                             end_range=config.max_layer_repeats + 1,
                                             mutation_chance=config.layer_repeat_mutation_chance if config.max_layer_repeats > 1 else 0)

    def get_all_mutagens(self):
        node_muts = super().get_all_mutagens()
        node_muts.extend([self.layer_type, self.activation, self.layer_repeats])
        return node_muts

    def convert_node(self, **kwargs) -> Tuple[Layer, Layer]:
        bp_node_id = kwargs['node_id']
        name = str(bp_node_id) if bp_node_id >= 0 else "agg(" + str(-1 * bp_node_id) + ")"
        name += "_" + (str(self.id) if self.id >= 0 else "agg(" + str(-1 * self.id) + ")")

        feature_multiplier = kwargs["feature_multiplier"]
        pheno = Layer(self, name, feature_multiplier=feature_multiplier)

        last_layer: Layer = pheno

        for i in range(self.layer_repeats.value - 1):
            if self.id < 0:
                raise Exception("cannot repeat agg layer")

            repeat_node_name = name + "_r" + repr(i)
            new_layer = Layer(self, repeat_node_name, feature_multiplier=feature_multiplier)
            last_layer.add_child(repeat_node_name, new_layer)
            last_layer = new_layer

        return pheno, last_layer

    def is_conv(self):
        return self.layer_type.value == nn.Conv2d

    def is_linear(self):
        return self.layer_type.value == nn.Linear

    def is_depthwise_sep(self):
        return self.layer_type.value == DepthwiseSeparableConv


def get_new_conv_parameter_mutagens():
    return {
        "conv_window_size": Option("conv_window_size", 1, 3, 5, 7, current_value=random.choice([1, 3, 5, 7]),
                                   mutation_chance=0.13),

        "conv_stride": IntegerVariable("conv_stride", current_value=1, start_range=1, end_range=5, mutation_chance=0.1),

        "reduction": Option("reduction", None, nn.MaxPool2d  # , nn.AvgPool2d
                            ,
                            current_value=nn.MaxPool2d if random.random() < config.module_node_max_pool_chance else None,
                            submutagens=
                            {
                                nn.MaxPool2d: {
                                    "pool_size": IntegerVariable("pool_size",
                                                                 current_value=2,
                                                                 start_range=2,
                                                                 end_range=5,
                                                                 mutation_chance=0.1)}

                                # ,nn.AvgPool2d: {
                                #     "pool_size": IntegerVariable("pool_size",
                                #                                  current_value=2,
                                #                                  start_range=2,
                                #                                  end_range=5,
                                #                                  mutation_chance=0.1)}
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

        "out_features": IntegerVariable("out_features", current_value=int(random.normalvariate(mu=50, sigma=20)),
                                        start_range=1,
                                        end_range=100, mutation_chance=0.22),

        "pad_output": Option("pad_output", True, False,
                             current_value=random.choices([True, False], weights=[0.65, 0.35])[0])
    }


def get_new_linear_parameter_mutagens():
    return {
        "regularisation": Option("regularisation", None, nn.BatchNorm1d,
                                 current_value=nn.BatchNorm1d if random.random() < config.module_node_batchnorm_chance else None,
                                 mutation_chance=0.15),

        "dropout": Option("regularisation", None, nn.Dropout,
                          current_value=nn.Dropout if random.random() < config.module_node_dropout_chance else None,
                          submutagens=
                          {
                              nn.Dropout: {
                                  "dropout_factor": ContinuousVariable("dropout_factor", current_value=0.15,
                                                                       start_range=0, end_range=0.75,
                                                                       mutation_chance=0.15)
                              }
                          }, mutation_chance=0.08),

        "out_features": IntegerVariable("out_features", current_value=int(random.normalvariate(mu=200, sigma=50)),
                                        start_range=10,
                                        end_range=1024, mutation_chance=0.22)
    }


def get_new_regulariser_only_parameter_mutagens():
    return {
        "regularisation": Option("regularisation", None, 'batchnorm',
                                 current_value='batchnorm' if random.random() < config.module_node_batchnorm_chance else None,
                                 mutation_chance=0.15),

        "dropout": Option("regularisation", None, nn.Dropout,
                          current_value=nn.Dropout if random.random() < config.module_node_dropout_chance else None,
                          submutagens=
                          {
                              nn.Dropout: {
                                  "dropout_factor": ContinuousVariable("dropout_factor", current_value=0.15,
                                                                       start_range=0, end_range=0.75,
                                                                       mutation_chance=0.15)
                              }
                          }, mutation_chance=0.08),

        "out_features": IntegerVariable("out_features", current_value=int(random.normalvariate(mu=200, sigma=50)),
                                        start_range=10,
                                        end_range=1024, mutation_chance=0.22)
    }


def get_new_depthwise_conv_parameter_mutagens():
    conv_params = get_new_conv_parameter_mutagens()
    conv_params["kernels_per_layer"] = IntegerVariable("kernels_per_layer",
                                                       current_value=int(random.normalvariate(mu=15, sigma=2)),
                                                       start_range=1, end_range=50, mutation_chance=0.15)
    conv_params["conv_window_size"] = Option("conv_window_size", 3, 5, 7, current_value=random.choice([3, 5, 7]),
                                             mutation_chance=0.13)
    return conv_params


def get_new_layer_submutagens():
    subs = {
        nn.Conv2d: get_new_conv_parameter_mutagens(),
        nn.Linear: get_new_linear_parameter_mutagens(),
        None: get_new_regulariser_only_parameter_mutagens()
    }
    if config.use_depthwise_separable_convs:
        subs[DepthwiseSeparableConv] = get_new_depthwise_conv_parameter_mutagens()

    return subs
