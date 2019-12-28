from __future__ import annotations

import os
from functools import reduce
from typing import List, Union, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn, tensor, optim, squeeze

from runs.runs_manager import get_fully_train_folder_path
from src2.configuration import config
from src2.phenotype.neural_network.layers.layer import Layer
from src2.analysis.visualisation import phenotype_visualiser
from src2.analysis.visualisation.phenotype_visualiser import get_node_colour

if TYPE_CHECKING:
    from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
    from src2.phenotype.neural_network.layers.aggregation_layer import AggregationLayer


class Network(nn.Module):
    def __init__(self, blueprint: BlueprintGenome, input_shape: list, output_dim=10, sample_map=None):
        """
        Constructs a trainable nn.Module network given a Blueprint genome. Must have access to a generation singleton
        with a module population.

        :param blueprint: The blueprint used to construct the network
        :param input_shape: The shape of the input
        :param output_dim: The required dimension of the output (assumed to be 1D)
        :param sample_map: Required to construct a network with specific modules from each species (usually used when fully training)
        """
        super().__init__()
        self.blueprint: BlueprintGenome = blueprint
        self.output_dim = output_dim

        self.model: Layer
        (self.model, output_layer), self.sample_map = blueprint.to_phenotype(sample_map=sample_map)
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
        suffix = ("_p" + str(parse_number) if parse_number >= 0 else "")
        phenotype_visualiser.visualise(self, prefix, suffix, node_colour=get_node_colour)

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_location(self) -> str:
        return os.path.join(get_fully_train_folder_path(config.run_name), 'bp-' + str(self.blueprint.id) + '.model')

    def save(self):
        if not os.path.exists(get_fully_train_folder_path(config.run_name)):
            os.makedirs(get_fully_train_folder_path(config.run_name))

        torch.save(self.state_dict(), self.save_location())

    def load(self):
        self.load_state_dict(torch.load(self.save_location()))
