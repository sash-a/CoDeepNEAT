from __future__ import annotations

from functools import reduce
from typing import List, Union, Tuple, TYPE_CHECKING

import graphviz
import torch.nn.functional as F
from torch import nn, tensor, optim, squeeze

from runs import RunsManager
from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer
from src2.Phenotype.NeuralNetwork.Layers.BaseLayer import BaseLayer
from src2.Phenotype.NeuralNetwork.Layers.Layer import Layer

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome


class Network(nn.Module):
    def __init__(self, blueprint: BlueprintGenome, input_shape: list, output_dim=10, prescribed_sample_map = None, **kwargs):
        super().__init__()
        self.blueprint: BlueprintGenome = blueprint
        self.output_dim = output_dim

        self.model: Layer
        (self.model, output_layer), self.sample_map = blueprint.to_phenotype(prescribed_sample_map = prescribed_sample_map, **kwargs)

        self.shape_layers(input_shape)

        # shaping the final layer
        img_flat_size = int(reduce(lambda x, y: x * y, output_layer.out_shape) / output_layer.out_shape[0])
        self.final_layer = nn.Linear(img_flat_size, output_dim)

        self.loss_fn = nn.NLLLoss()  # TODO mutagen
        self.optimizer: optim.adam = optim.Adam(self.parameters(), lr=self.blueprint.learning_rate.value,
                                                betas=(self.blueprint.beta1.value, self.blueprint.beta2.value))

    def forward(self, x):
        q: List[Tuple[Union[Layer, AggregationLayer], tensor]] = [(self.model, x)]

        while q:
            layer, x = q.pop()
            x = layer(x)
            # input will be None if agg layer has not received all its inputs yet
            if x is not None:
                q.extend([(child, x) for child in list(layer.child_layers)])

        # TODO final activation function should be evolve-able
        batch_size = x.size()[0]
        final_layer_out = F.relu(self.final_layer(x.view(batch_size, -1)))
        return squeeze(F.log_softmax(final_layer_out.view(batch_size, self.output_dim, -1), dim=1))

    def shape_layers(self, in_shape: list):
        q: List[Tuple[Union[Layer, AggregationLayer], list]] = [(self.model, in_shape)]

        while q:
            layer, input_shape = q.pop()
            output_shape = layer.create_layer(input_shape)

            if output_shape is not None:
                q.extend([(child, output_shape) for child in list(layer.child_layers)])

    def multiply_learning_rate(self, factor):
        pass

    def visualize(self, parse_number=-1, prefix=""):
        name = prefix + "blueprint_i" + str(self.blueprint.id) + "_phenotype" + (
            "_p" + str(parse_number) + "_" if parse_number >= 0 else "")
        # print("saving:", name, "to",RunsManager.get_graphs_folder_path(config.run_name))
        graph = graphviz.Digraph(name=name, comment='Phenotype')

        q: List[BaseLayer] = [self.model]
        graph.node(self.model.name, self.model.get_layer_info())
        visited = set()

        while q:
            parent_layer = q.pop()

            if parent_layer.name not in visited:
                visited.add(parent_layer.name)
                for child_layer in parent_layer.child_layers:
                    description = child_layer.get_layer_info()

                    graph.node(child_layer.name, child_layer.name + '\n' + description)
                    graph.edge(parent_layer.name, child_layer.name)

                    q.append(child_layer)

        graph.render(quiet=False, directory=RunsManager.get_graphs_folder_path(config.run_name), view=config.view_graph_plots)
