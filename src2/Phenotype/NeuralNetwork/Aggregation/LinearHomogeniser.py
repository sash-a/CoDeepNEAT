from typing import List

from torch import tensor
import torch.nn.functional as F


def homogenise(linear_inputs: List[tensor]) -> List[tensor]:
    """Homogenises linear layers for merging using sum"""
    max_size = max([list(linear_input.size())[1] for linear_input in linear_inputs])

    for i in range(len(linear_inputs)):
        linear_inputs[i] = F.pad(input=linear_inputs[i], pad=[0, max_size - list(linear_inputs[i].size())[1]])

    return linear_inputs

