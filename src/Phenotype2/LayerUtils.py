from abc import ABC

from torch import nn, tensor


class BaseLayer(nn.Module, ABC):
    pass


class Reshape(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size: tuple = size

    def forward(self, inp: tensor):
        return inp.reshape(*self.size)
