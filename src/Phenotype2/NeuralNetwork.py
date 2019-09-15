from torch import nn, tensor, optim

from src.Config import Config
from src.CoDeepNEAT.CDNGenomes.BlueprintGenome import BlueprintGenome
from src.NEAT.Species import Species

from src.Phenotype2.Layer import Layer
from src.Phenotype2.AggregationLayer import AggregationLayer
from src.Phenotype2.LayerUtils import BaseLayer, Reshape

from functools import reduce
from typing import List, Union, Tuple


class Network(nn.Module):
    def __init__(self, blueprint: BlueprintGenome, module_species: List[Species], input_shape: list, output_dim=10):
        super().__init__()
        self.blueprint: BlueprintGenome = blueprint

        self.model, output_layer = blueprint.to_phenotype(None, module_species)
        self.shape_layers(input_shape)

        print(output_layer.out_shape)
        # shaping the final layer
        img_flat_size = int(reduce(lambda x, y: x * y, output_layer.out_shape) / output_layer.out_shape[0])
        self.use_final_reshape = False
        if len(output_layer.out_shape) != 2 or output_layer.out_shape[1] != img_flat_size:
            self.use_final_reshape = True
            self.reshape_layer = Reshape(input_shape[0], img_flat_size)

        self.final_layer = nn.Linear(img_flat_size, output_dim)

        self.loss_fn = nn.NLLLoss()
        self.optimizer: optim.adam = optim.Adam(self.parameters(), lr=self.blueprint.learning_rate.value,
                                                betas=(self.blueprint.beta1.value, self.blueprint.beta2.value))

    def forward(self, input):
        q: List[Tuple[Union[Layer, AggregationLayer], tensor]] = [(self.model, input)]

        while q:
            layer, input = q.pop()
            input = layer(input)
            # input will be None if agg layer has not received all its inputs yet
            if input is not None:
                if Config.use_graph:
                    q.extend([(child, input) for child in list(layer.child_layers)])
                else:
                    q.extend([(child, input) for child in list(layer.children()) if isinstance(child, BaseLayer)])

        if self.use_final_reshape:
            return self.final_layer(self.reshape_layer(input))
        else:
            return self.final_layer(input)

    def shape_layers(self, in_shape: list):
        q: List[Tuple[Union[Layer, AggregationLayer], list]] = [(self.model, in_shape)]

        while q:
            layer, input_shape = q.pop()
            output_shape = layer.create_layer(input_shape)
            # out_shape will be None if agg layer has not received all its inputs yet
            if output_shape is not None:
                if Config.use_graph:
                    q.extend([(child, output_shape) for child in list(layer.child_layers)])
                else:
                    q.extend(
                        [(child, output_shape) for child in list(layer.children()) if isinstance(child, BaseLayer)])

    def multiply_learning_rate(self, factor):
        pass
