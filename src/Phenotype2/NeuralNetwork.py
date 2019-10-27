import graphviz
from torch import nn, tensor, optim, squeeze
import torch.nn.functional as F

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
        self.output_dim = output_dim

        self.model: Layer
        self.model, output_layer = blueprint.to_phenotype(None, module_species)
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
                if Config.use_graph:
                    q.extend([(child, x) for child in list(layer.child_layers)])
                else:
                    q.extend([(child, x) for child in list(layer.children()) if isinstance(child, BaseLayer)])

        # TODO final activation function should be evolvable
        # img_flat_size = int(reduce(lambda a, b: a * b, list(x.size())) / list(x.size())[0])
        batch_size = x.size()[0]
        final_layer_out = F.relu(self.final_layer(x.view(batch_size, -1)))
        return squeeze(F.log_softmax(final_layer_out.view(batch_size, self.output_dim, -1), dim=1))

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

    def visualize(self, filename='new', view=False):
        graph = graphviz.Digraph(name='New graph', comment='New graph', filename=filename)

        q: List[BaseLayer] = [self.model]
        graph.node(self.model.name, self.model.get_layer_info())
        visited = set()

        while q:
            layer = q.pop()

            if layer.name not in visited:
                visited.add(layer.name)
                for child in layer.child_layers:
                    description = child.get_layer_type_name()

                    graph.node(child.name, description)
                    graph.edge(layer.name, child.name)

                    q.append(child)

        graph.render(filename, view=view)
