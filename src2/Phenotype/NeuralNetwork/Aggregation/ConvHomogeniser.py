from __future__ import annotations

from typing import List, TYPE_CHECKING

from torch import tensor, nn
import torch.nn.functional as F

from src2.Configuration import config

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer


def homogenise_xy(conv_inputs: List[tensor]) -> List[tensor]:
    """Homogenises the x, y dims of a conv output. Assumes that the output is square i.e x = y"""
    for i in range(len(conv_inputs)):  # Checks that all inputs are square
        _, _, x, y = list(conv_inputs[i].size())
        if x != y:
            conv_inputs[i] = pad_to_square(conv_inputs[i].size())

    # Try to get all tensors as close to the average size of each tensor passed in
    target_size = round(sum([list(conv_input.size())[2] for conv_input in conv_inputs]) / len(conv_inputs))

    for i in range(len(conv_inputs)):  # downsize all tensors larger than mean to be as close to mean as possible
        size = list(conv_inputs[i].size())[2]
        if size > target_size:
            pooling_factor = round(size / target_size)
            if pooling_factor > 1:
                conv_inputs[i] = F.max_pool2d(conv_inputs[i], kernel_size=(pooling_factor, pooling_factor))

    # Some tensors may still be larger than the average size, therefore the new target is the size of the largest tensor
    target_size = max([list(conv_input.size())[2] for conv_input in conv_inputs])

    for i in range(len(conv_inputs)):  # downsize all tensors larger than mean to be as close to mean as possible
        size = list(conv_inputs[i].size())[2]
        left_pad = (target_size - size) // 2
        right_pad = (target_size - size) - left_pad
        conv_inputs[i] = F.pad(conv_inputs[i], [left_pad, right_pad, left_pad, right_pad])

    return conv_inputs


def homogenise_channel(conv_inputs: List[tensor], agg_layer: AggregationLayer) -> List[tensor]:
    """This will only be used when merging using a lossy strategy"""
    # TODO test!
    if not agg_layer.channel_resizers:  # If 1x1 convs not yet created then create them
        # print('No 1x1 convs found for channel resizing, creating them')
        target_size = round(sum([list(conv_input.size())[1] for conv_input in conv_inputs]) / len(conv_inputs))

        for conv_input in conv_inputs:  # creating 1x1 convs
            channel = list(conv_input.size())[1]
            agg_layer.channel_resizers.append(nn.Conv2d(channel, target_size, 1))

        agg_layer.channel_resizers.to(config.get_device())

    for i in range(len(conv_inputs)):  # passing inputs through 1x1 convs
        # print('using 1x1 conv for passing input through an agg node with ', len(agg_layer.inputs), 'inputs')
        conv_inputs[i] = agg_layer.channel_resizers[i](conv_inputs[i])

    # print('done passing through 1x1s')
    return conv_inputs


def pad_to_square(conv_input: tensor) -> tensor:
    _, _, x, y = list(conv_input.size())
    if x > y:
        left = (x - y) // 2
        right = x - y - left
        conv_input = F.pad(conv_input, [left, right])
    else:
        left = (y - x) // 2
        right = y - x - left
        conv_input = F.pad(conv_input, [0, 0, left, right])

    return conv_input
