from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from torch import nn

from src.Config import Config


class BaseLayer(nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.out_shape: List[int] = []
        self.child_layers: List[BaseLayer] = []

    def add_child(self, name: str, child: BaseLayer) -> None:
        if Config.use_graph:
            self.child_layers.append(child)

        self.add_module(name, child)

    # child_layers = property(lambda self: [child for child in self.children() if isinstance(child, BaseLayer)])

    @abstractmethod
    def create_layer(self, in_shape) -> List[int]:
        pass

    @abstractmethod
    def get_layer_info(self) -> str:
        pass
