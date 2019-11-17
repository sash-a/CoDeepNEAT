from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union, TYPE_CHECKING
from torch import nn

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer
    from src2.Phenotype.NeuralNetwork.Layers.Layer import Layer


class BaseLayer(nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.out_shape: List[int] = []
        self.child_layers: List[BaseLayer] = []

    def add_child(self, name: str, child: BaseLayer) -> None:
        self.add_module(name, child)
        self.child_layers.append(child)

    # child_layers: List[Union[Layer, BaseLayer, AggregationLayer]] = \
    #     property(lambda self: [child for child in self.children() if isinstance(child, BaseLayer)])

    @abstractmethod
    def create_layer(self, in_shape) -> List[int]:
        pass

    @abstractmethod
    def get_layer_info(self) -> str:
        pass
