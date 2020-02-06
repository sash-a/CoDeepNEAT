from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    pass


class BaseLayer(nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.child_layers: List[BaseLayer] = []

    def add_child(self, name: str, child: BaseLayer) -> None:
        self.add_module(name, child)
        self.child_layers.append(child)

    # child_layers: List[Union[Layer, BaseLayer, AggregationLayer]] = \
    #     property(lambda self: [child for child in self.children() if isinstance(child, BaseLayer)])

    @abstractmethod
    def create_layer(self, in_shape, feature_multiplier=1) -> List[int]:
        pass

    @abstractmethod
    def get_layer_info(self) -> str:
        pass
