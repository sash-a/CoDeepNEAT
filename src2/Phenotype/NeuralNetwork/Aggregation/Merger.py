from __future__ import annotations

from typing import List, TYPE_CHECKING

import torch
from torch import tensor

from src2.Phenotype.NeuralNetwork.Aggregation import ConvHomogeniser, LinearHomogeniser, ConvLinearHomogeniser

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer


def merge(inputs: List[tensor], agg_layer: AggregationLayer) -> tensor:
    linear_inputs = [x for x in inputs if len(list(x.size())) == 2]
    conv_inputs = [x for x in inputs if len(list(x.size())) == 4]

    lossy = agg_layer.lossy

    if linear_inputs and conv_inputs:
        # print('both inputs')
        linear = merge_layers(homogenise_linear(linear_inputs, lossy), lossy) if len(linear_inputs) > 1 else \
            linear_inputs[0]
        # print('linears merged, shape: ', linear.size(), "numel:", linear.numel())
        conv = merge_layers(homogenise_conv(conv_inputs, agg_layer), lossy) if len(conv_inputs) > 1 else conv_inputs[0]
        # print('convs merged, shape:', conv.size(), "numel:", conv.numel())

        if agg_layer.try_output_conv:
            lossy = False  # cannot do lossy merging of a conv produced from a linear and another conv
            # print('constructing conv from linear')
            conv_construct = ConvLinearHomogeniser.reshape_linear_to_conv(linear, conv)
            # print('conv constructed')
            homos = ConvHomogeniser.homogenise_xy([conv, conv_construct])
            # print('convs homogenised')
        else:
            # print('flattening conv')
            flat_conv = ConvLinearHomogeniser.flatten_conv(conv)
            # print('conv flattened')
            homos = homogenise_linear([flat_conv, linear], lossy)
            # print('linears homogenised')

    elif linear_inputs and not conv_inputs:
        # print('only linears')
        homos = homogenise_linear(inputs, lossy)
        # print('linears homogenised')

    elif conv_inputs and not linear_inputs:
        # print('only convs')
        homos = homogenise_conv(inputs, agg_layer)
        # print('convs homogenised')

    else:
        raise Exception("erroneous or empty inputs passed to agg layer")

    # print('done merging...')
    return merge_layers(homos, lossy)


def merge_layers(homogeneous_inputs: List[tensor], lossy: bool) -> tensor:
    if lossy:
        return torch.sum(torch.stack(homogeneous_inputs), dim=0)
    else:
        return torch.cat(homogeneous_inputs, dim=1)


def homogenise_conv(inputs: List[tensor], agg_layer: AggregationLayer):
    homos = ConvHomogeniser.homogenise_xy(inputs)
    if agg_layer.lossy:
        homos = ConvHomogeniser.homogenise_channel(homos, agg_layer)

    return homos


def homogenise_linear(inputs: List[tensor], lossy: bool):
    if lossy:
        return LinearHomogeniser.homogenise(inputs)
    else:
        return inputs
