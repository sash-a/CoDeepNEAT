import torch
from torch import nn
import math
from functools import reduce

from src.CoDeepNEAT.CDNNodes.ModuleNode import ModuleNEATNode
from src.Phenotype2.LayerUtils import Reshape, BaseLayer
import src.Config.Config as Config


class Layer(BaseLayer):
    def __init__(self, module: ModuleNEATNode, feature_multiplier=1):
        super().__init__()
        self.module_node: ModuleNEATNode = module

        self.out_features = round(module.layer_type.get_sub_value('out_features') * feature_multiplier)
        print('of', self.out_features)

        self.deep_layer: nn.Module = None  # layer does not yet have a size
        self.reshape_layer: nn.Module = None
        self.regularisation: nn.Module = None
        self.reduction: nn.Module = None
        self.dropout: nn.Module = None

        self.activation: nn.Module = module.activation.value  # to device?

        neat_regularisation = module.layer_type.get_sub_value('regularisation', return_mutagen=True)
        neat_reduction = module.layer_type.get_sub_value('reduction', return_mutagen=True)
        neat_dropout = module.layer_type.get_sub_value('dropout', return_mutagen=True)

        device = Config.get_device()

        if neat_regularisation.value is not None:
            self.regularisation = neat_regularisation()(self.out_features).to(device)

        if neat_reduction is not None and neat_reduction.value is not None:
            if neat_reduction.value == nn.MaxPool2d or neat_reduction.value == nn.MaxPool1d:
                pool_size = neat_reduction.get_sub_value('pool_size')
                if neat_reduction.value == nn.MaxPool2d:
                    self.reduction = nn.MaxPool2d(pool_size, pool_size).to(device)  # TODO this should be stride
                elif neat_reduction.value == nn.MaxPool1d:
                    # TODO should be this: but need to calc size for 1d nn.MaxPool1d(pool_size).to(device)
                    self.reduction = nn.MaxPool2d(pool_size, pool_size).to(device)
            else:
                raise Exception('Error unimplemented reduction ' + repr(neat_reduction()))

        if neat_dropout is not None and neat_dropout.value is not None:
            self.dropout = neat_dropout.value(neat_dropout.get_sub_value('dropout_factor')).to(device)

    def forward(self, x):
        # TODO layer sizing
        if self.reshape_layer is not None:
            x = self.reshape_layer(x)
        if self.deep_layer is not None:
            x = self.deep_layer(x)
        if self.regularisation is not None:
            x = self.regularisation(x)
        if self.reduction is not None:
            x = self.reduction(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def create_layer(self, in_shape: list):
        if len(in_shape) == 4:
            batch, channels, h, w = in_shape
        elif len(in_shape) == 2:
            batch, channels = in_shape
        else:
            raise Exception('Invalid input of size ' + str(len(in_shape)))

        flat_size = int(reduce(lambda x, y: x * y, in_shape))

        # Calculating out feature size, creating deep layer and reshaping if necessary
        if self.module_node.layer_type.value == nn.Conv2d:
            if len(in_shape) == 2:
                # TODO non-square
                h = w = int(math.sqrt(flat_size / channels))
                self.reshape_layer = Reshape(batch, channels, h, w).to(Config.get_device())

            # TODO padding the image could help
            # Also could make kernel size and stride a tuple
            padding = 0
            dilation = 1
            kernel_size = self.module_node.layer_type.get_sub_value('conv_window_size')
            stride = self.module_node.layer_type.get_sub_value('conv_stride')
            h_out = math.floor((h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            w_out = math.floor((w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

            # If using max pooling out feature size changes
            neat_reduction = self.module_node.layer_type.get_sub_value('reduction', return_mutagen=True)
            if neat_reduction is not None and neat_reduction.value is not None:  # Using max pooling
                pool_size = neat_reduction.get_sub_value('pool_size')
                h_out = (h_out - pool_size) / pool_size + 1
                w_out = (w_out - pool_size) / pool_size + 1

            self.deep_layer = nn.Conv2d(channels, self.out_features, kernel_size, stride).to(Config.get_device())
            return [batch, self.out_features, h_out, w_out]
        else:  # self.module_node.layer_type.value == nn.Linear:
            if len(in_shape) != 2 or channels != flat_size:
                self.reshape_layer = Reshape(batch, flat_size).to(Config.get_device())

            self.deep_layer = nn.Linear(flat_size, int(self.out_features)).to(Config.get_device())
            return [batch, self.out_features]
# TODO
# ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 121])
