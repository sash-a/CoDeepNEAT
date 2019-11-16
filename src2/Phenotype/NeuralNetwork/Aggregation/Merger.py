from __future__ import annotations

from typing import List, TYPE_CHECKING

import torch

from src2.Phenotype.NeuralNetwork.Aggregation import ConvHomogeniser, LinearHomogeniser, ConvLinearHomogeniser

from torch import tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer


def merge(inputs: List[tensor], agg_layer: AggregationLayer) -> tensor:
    linear_inputs = [x for x in inputs if len(list(x.size())) == 2]
    conv_inputs = [x for x in inputs if len(list(x.size())) == 4]

    lossy = agg_layer.lossy

    if linear_inputs and conv_inputs:
        linear = merge_layers(homogenise_linear(linear_inputs, lossy), lossy)
        conv = merge_layers(homogenise_conv(linear_inputs, agg_layer), lossy)

        if agg_layer.try_output_conv:
            lossy = False  # cannot do lossy merging of a conv produced from a linear and another conv
            conv_construct = ConvLinearHomogeniser.reshape_linear_to_conv(linear, conv)
            homos = ConvHomogeniser.homogenise_xy([conv, conv_construct])

        else:
            flat_conv = ConvLinearHomogeniser.flatten_conv(conv)
            homos = homogenise_linear([flat_conv, linear], lossy)

    elif linear_inputs and not conv_inputs:
        homos = homogenise_linear(inputs, lossy)

    elif conv_inputs and not linear_inputs:
        homos = homogenise_conv(inputs, agg_layer)

    else:
        raise Exception("erroneous or empty inputs passed to agg layer")

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
