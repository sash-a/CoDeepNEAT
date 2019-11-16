import math
from typing import List

from torch import tensor
from torch import nn
import torch.nn.functional as F


def homogenise_xy(conv_inputs: List[tensor]):
    """Homogenises the x, y dims of a conv output. Assumes that the output is square i.e x = y"""
    for conv_input in conv_inputs:  # Checks that all inputs are square
        _, _, x, y = list(conv_input.size())
        if x != y:
            pad_to_square(conv_input)

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


def homogenise_channel(conv_inputs: List[tensor]):
    pass


def pad_to_square(conv_input: tensor):
    pass


import torch

l = [
    torch.randn(5, 4, 10, 10),
    torch.randn(5, 4, 20, 20),
    torch.randn(5, 4, 50, 50),
    torch.randn(5, 4, 5, 5)
]

homogenise_xy(l)
print([list(conv_input.size()) for conv_input in l])
