from torch import nn, tensor
import torch

from src.CoDeepNEAT.CDNGenomes.BlueprintGenome import BlueprintGenome
from src.NEAT.Species import Species

from src.Phenotype2.Layer import Layer
from src.Phenotype2.AggregationLayer import AggregationLayer
from src.Phenotype2.LayerUtils import BaseLayer, Reshape

from src.Config import Config

from functools import reduce
from typing import List, Union


class Network(nn.Module):
    def __init__(self, blueprint: BlueprintGenome, module_species: List[Species], input_shape: list, output_dim=10):
        super().__init__()
        self.blueprint: BlueprintGenome = blueprint

        self.model, output_layer = blueprint.to_phenotype(None, module_species)
        final_shape = self.shape_layers(self.model, input_shape, output_layer)

        # shaping the final layer
        flat_size = reduce(lambda x, y: x * y, final_shape)
        self.reshape = False
        if len(final_shape) != 2 or final_shape[1] != flat_size:
            self.reshape = True
            self.reshape_layer = Reshape(input_shape[0], flat_size).to(Config.get_device())

        self.final_layer = nn.Linear(flat_size, output_dim).to(Config.get_device())

        # TODO: get params and add optimizer

    def forward(self, input):
        if self.reshape:
            return self.final_layer(self.reshape_layer(self.model(input)))
        return self.final_layer(self.model(input))

    def shape_layers(self, layer: Union[Layer, AggregationLayer], in_shape: list, output_layer):
        out_shape = layer.create_layer(in_shape)
        if layer == output_layer:
            return out_shape

        for child in layer.children():
            if isinstance(child, BaseLayer):
                self.shape_layers(child, out_shape, output_layer)

        return out_shape


from NEAT.Gene import ConnectionGene
from CoDeepNEAT.CDNGenomes.ModuleGenome import ModuleGenome
from CoDeepNEAT.CDNNodes.BlueprintNode import BlueprintNEATNode
from CoDeepNEAT.CDNNodes.ModuleNode import ModuleNEATNode, NodeType

conn0 = ConnectionGene(0, 0, 2)
conn1 = ConnectionGene(1, 0, 3)
conn2 = ConnectionGene(2, 2, 5)
conn3 = ConnectionGene(3, 3, 6)
conn4 = ConnectionGene(4, 6, 4)
conn5 = ConnectionGene(5, 5, 1)
conn6 = ConnectionGene(6, 4, 1)
conn7 = ConnectionGene(7, 3, 5)

# conn3.enabled.set_value(False)

n0 = ModuleNEATNode(0, NodeType.INPUT)
n1 = ModuleNEATNode(1, NodeType.OUTPUT)
n2 = ModuleNEATNode(2)
n3 = ModuleNEATNode(3)
n4 = ModuleNEATNode(4)
n5 = ModuleNEATNode(5)
n6 = ModuleNEATNode(6)
genome0 = ModuleGenome([conn0, conn1, conn2, conn3, conn4, conn5, conn6, conn7], [n0, n1, n2, n3, n4, n5, n6])

conn10 = ConnectionGene(0, 0, 1)
n10 = ModuleNEATNode(0, NodeType.INPUT)
n11 = ModuleNEATNode(1, NodeType.OUTPUT)
n10.layer_type.set_value(nn.Linear)
genome1 = ModuleGenome([conn10], [n10, n11])

# in_layer, out_layer = genome0.to_phenotype(None)
# print(in_layer)

spcs = [Species(genome0), Species(genome1)]

bn0 = BlueprintNEATNode(0, NodeType.INPUT)
bn1 = BlueprintNEATNode(1, NodeType.OUTPUT)

bn0.species_number.set_value(1)
bn1.species_number.set_value(1)

bpg = BlueprintGenome([conn10], [bn0, bn1])

import src.Validation.DataLoader as DL
import math
from src.Utilities.Utils import get_flat_number

x: tensor
x, target = DL.sample_data(Config.get_device(), 16)
enen = Network(bpg, spcs, list(x.shape))
print(enen)
print(enen(x))

# h_out = (32 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1
# print(h_out)
#
# # x = x.reshape(x.shape[0], -1)
# # lin = nn.Linear(3 * 32 * 32, 10).to(Cfg.get_device())
# # x = lin(x)
# # print(x.size())
#
# conv_dim = int(math.sqrt(10 / 10))
# # x = x.reshape([1, 10, 1, 1])
# conv = nn.Conv2d(3, 23513, 3, 1).to(Cfg.get_device())
# x = conv(x)
# print(x.size())
#
# bn = nn.BatchNorm2d(23513).to(Cfg.get_device())
# bn(x).size()
