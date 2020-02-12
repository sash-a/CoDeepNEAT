from __future__ import annotations

from typing import List, TYPE_CHECKING

import torch
from torch import tensor

from src.phenotype.neural_network.aggregation import conv_homogeniser, linear_homogeniser, conv_linear_homogeniser

if TYPE_CHECKING:
    from src.phenotype.neural_network.layers.aggregation_layer import AggregationLayer


def merge(inputs: List[tensor], agg_layer: AggregationLayer) -> tensor:
    linear_inputs = [x for x in inputs if len(list(x.size())) == 2]
    conv_inputs = [x for x in inputs if len(list(x.size())) == 4]

    lossy = agg_layer.lossy
    multiplication = agg_layer.use_element_wise_multiplication
    try_output_conv = agg_layer.try_output_conv

    if multiplication:
        # multiplicative agg requires tensors to match, and gets preference
        print("turning on lossy to use mult")
        lossy = True
        try_output_conv = False

    if linear_inputs and conv_inputs:
        # print('both inputs')
        linear = _merge_layers(homogenise_linear(linear_inputs, lossy), lossy, multiplication) if len(linear_inputs) > 1 else \
            linear_inputs[0]
        # print('linears merged, shape: ', linear.size(), "numel:", linear.numel())
        conv = _merge_layers(homogenise_conv(conv_inputs, agg_layer), lossy, multiplication) if len(conv_inputs) > 1 else conv_inputs[0]
        # print('convs merged, shape:', conv.size(), "numel:", conv.numel())

        if try_output_conv:
            lossy = False  # cannot do lossy merging of a conv produced from a linear and another conv
            # print('constructing conv from linear')
            conv_construct = conv_linear_homogeniser.reshape_linear_to_conv(linear, conv)
            # print('conv constructed')
            homos = conv_homogeniser.homogenise_xy([conv, conv_construct])
            # print('convs homogenised')
        else:
            # print('flattening conv')
            flat_conv = conv_linear_homogeniser.flatten_conv(conv)
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
    return _merge_layers(homos, lossy, multiplication)


def _merge_layers(homogeneous_inputs: List[tensor], lossy: bool, multiply: bool) -> tensor:
    if multiply and not lossy:
        raise Exception("cannot do non lossy multiplication")
    if multiply:
        print("outs:",torch.stack(homogeneous_inputs).size())
        out = homogeneous_inputs[0]
        for i in range(1,len(homogeneous_inputs)):
            try:
                out = out * homogeneous_inputs[i]
            except Exception as e:
                print("out:", out.size())
                print("hom:",homogeneous_inputs[i].size())
                raise e
        return out
    elif lossy:
        return torch.sum(torch.stack(homogeneous_inputs), dim=0)
    else:
        return torch.cat(homogeneous_inputs, dim=1)


def homogenise_conv(inputs: List[tensor], agg_layer: AggregationLayer):
    homos = conv_homogeniser.homogenise_xy(inputs)
    if agg_layer.lossy:
        homos = conv_homogeniser.homogenise_channel(homos, agg_layer)

    return homos


def homogenise_linear(inputs: List[tensor], lossy: bool):
    if lossy:
        return linear_homogeniser.homogenise(inputs)
    else:
        return inputs
