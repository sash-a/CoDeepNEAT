from typing import Type, Union, Optional

from torch import nn, zeros
import math
from functools import reduce

from torch.nn import Sequential

from src.CoDeepNEAT.CDNNodes.ModuleNode import ModuleNEATNode
from src.Phenotype2.LayerUtils import Reshape, BaseLayer


class Layer(BaseLayer):
    def __init__(self, module: ModuleNEATNode, name, feature_multiplier=1):
        super().__init__(name)
        self.module_node: ModuleNEATNode = module

        self.out_features = round(module.layer_type.get_sub_value('out_features') * feature_multiplier)
        self.sequential: Optional[nn.Sequential] = None
        # TODO make these nn.Module activations so they can be added to the sequential
        self.activation: Optional[nn.Module] = self.module_node.activation.value

    def forward(self, x):
        return self.activation(self.sequential(x))

    def _create_regularisers(self):
        """Creates and returns regularisers given mutagens in self.module_node.layer_type"""
        regularisation: Optional[nn.Module] = None
        reduction: Optional[nn.Module] = None
        dropout: Optional[nn.Module] = None

        neat_regularisation = self.module_node.layer_type.get_sub_value('regularisation', return_mutagen=True)
        neat_reduction = self.module_node.layer_type.get_sub_value('reduction', return_mutagen=True)
        neat_dropout = self.module_node.layer_type.get_sub_value('dropout', return_mutagen=True)

        if neat_regularisation.value is not None and neat_regularisation.value is not None:
            regularisation = neat_regularisation()(self.out_features)

        if neat_reduction is not None and neat_reduction.value is not None:
            if neat_reduction.value == nn.MaxPool2d or neat_reduction.value == nn.MaxPool1d:
                pool_size = neat_reduction.get_sub_value('pool_size')
                if neat_reduction.value == nn.MaxPool2d:
                    reduction = nn.MaxPool2d(pool_size, pool_size)  # TODO should be stride
                elif neat_reduction.value == nn.MaxPool1d:
                    reduction = nn.MaxPool1d(pool_size)
            else:
                raise Exception('Error unimplemented reduction ' + repr(neat_reduction()))

        if neat_dropout is not None and neat_dropout.value is not None:
            dropout = neat_dropout.value(neat_dropout.get_sub_value('dropout_factor'))

        return tuple(r for r in [regularisation, reduction, dropout] if r is not None)

    def create_layer(self, in_shape: list):
        """
        Creates a layer of type nn.Linear or nn.Conv2d according to its module_node and gives it the correct shape.
        Populates the self.sequential attribute with created layers and values returned from self.create_regularisers.
        """
        if len(in_shape) == 4:
            batch, channels, h, w = in_shape
        elif len(in_shape) == 2:
            batch, channels = in_shape
        else:
            raise Exception('Invalid input with shape: ' + str(in_shape))

        reshape_layer: Optional[Reshape] = None
        img_flat_size = int(reduce(lambda x, y: x * y, in_shape) / batch)

        # Calculating out feature size, creating deep layer and reshaping if necessary
        if self.module_node.layer_type.value == nn.Conv2d:
            if len(in_shape) == 2:
                h = w = int(math.sqrt(img_flat_size / channels))
                reshape_layer = Reshape(batch, channels, h, w)

            # TODO make kernel size and stride a tuple
            window_size = self.module_node.layer_type.get_sub_value('conv_window_size')
            stride = self.module_node.layer_type.get_sub_value('conv_stride')
            padding = math.ceil((window_size - h) / 2)
            padding = padding if padding >= 0 else 0
            # dilation = 1

            # h_out = math.floor((h + 2 * padding - dilation * (window_size - 1) - 1) / stride + 1)
            # w_out = math.floor((w + 2 * padding - dilation * (window_size - 1) - 1) / stride + 1)  # same cause square
            #
            # # If using max pooling out feature size changes
            # neat_reduction = self.module_node.layer_type.get_sub_value('reduction', return_mutagen=True)
            # if neat_reduction is not None and neat_reduction.value is not None:  # Using max pooling
            #     pool_size = neat_reduction.get_sub_value('pool_size')
            #     h_out = math.ceil((h_out - pool_size) / pool_size + 1)
            #     w_out = math.ceil((w_out - pool_size) / pool_size + 1)

            deep_layer = nn.Conv2d(channels, self.out_features, window_size, stride, padding)
        else:  # self.module_node.layer_type.value == nn.Linear:
            if len(in_shape) != 2 or channels != img_flat_size:
                reshape_layer = Reshape(batch, img_flat_size)

            deep_layer = nn.Linear(img_flat_size, self.out_features)

        self.sequential = nn.Sequential(*[module for module in
                                          [reshape_layer, deep_layer, *self._create_regularisers()]
                                          if module is not None])
        self.out_shape = list(self.forward(zeros(in_shape)).size())

        return self.out_shape

    def get_layer_info(self):
        """for dnn visualization"""

        layer_type = self.module_node.layer_type
        extras = ""
        if layer_type() == nn.Conv2d:
            extras += "\nwidow size:" + repr(self.deep_layer.kernel_size)
        extras += "\nout features:" + repr(self.out_features)
        extras += "\n" + repr(self.regularisation).split("(")[0] if not (self.regularisation is None) else ""
        extras += "\n" + repr(self.reduction).split("(")[0] if not (self.reduction is None) else ""
        extras += "\n" + repr(self.dropout).split("(")[0] if not (self.dropout is None) else ""

        if layer_type() == nn.Conv2d:
            return "Conv" + extras

        elif layer_type() == nn.Linear:
            return "Linear" + extras
        else:
            print("layer type", layer_type(), "not implemented")
