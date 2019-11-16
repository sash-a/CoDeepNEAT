from math import floor, ceil
from typing import List

from torch import tensor
import torch.nn.functional as F


def flatten_conv(conv_input: tensor) -> tensor:
    batch_size = list(conv_input.size())[0]
    return conv_input.view(batch_size, -1)


def reshape_linear_to_conv(linear_input: tensor, conv_input_stencil: tensor, ) -> tensor:
    """
        try reshape the linear, picking the channel count to make the x,y as close to the x,y of the conv as possible
        Note this will return a conv construct made from the linear which is similar but not identical to the given conv
        The two convs should be homogenised after
    """

    if list(conv_input_stencil.size())[2] != list(conv_input_stencil.size())[3]:
        raise Exception("non square conv input")

    linear_features = list(linear_input.size())[1]
    conv_xy = list(conv_input_stencil.size())[2] ** 2

    ratio = linear_features / conv_xy
    channel_size_options = [floor(ratio), ceil(ratio)]

    if ratio > 1:
        min_pad_value = 10000
        best_channel_option = -1
        for c in channel_size_options:
            adjusted_ratio = ratio / c
            pad_value = abs(1 - adjusted_ratio)
            if pad_value < min_pad_value:
                min_pad_value = pad_value
                best_channel_option = c
    else:
        best_channel_option = 1

    """
        together with c, x defines the size of the conv output the linear will be reshaped to.
        we ceil to ensure that this conv output >= linear input. then we pad the linear input: linear input = conv output
    """
    x = ceil(pow(linear_features / best_channel_option, 0.5))

    constructed_conv_features = x ** 2 * best_channel_option
    required_linear_pad = constructed_conv_features - linear_features
    left_pad = required_linear_pad // 2
    right_pad = required_linear_pad - left_pad
    linear_input = F.pad(input=linear_input, pad=[left_pad, right_pad])

    return linear_input.view(list(linear_input.size())[0], best_channel_option, x, x)
