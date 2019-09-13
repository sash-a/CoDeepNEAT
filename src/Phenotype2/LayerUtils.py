from abc import ABC, abstractmethod
from typing import List
from torch import nn, tensor


class BaseLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.out_shape: List[int] = []

    @abstractmethod
    def create_layer(self, in_shape):
        pass


class Reshape(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size: list = list(size)

    def forward(self, inp: tensor):
        # Allows for reshaping independent of batch size
        batch_size = list(inp.size())[0]
        if batch_size != self.size[0]:
            self.size[0] = batch_size

        return inp.reshape(*self.size)
