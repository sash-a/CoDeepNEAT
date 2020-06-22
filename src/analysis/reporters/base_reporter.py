from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

from src.phenotype.neural_network.neural_network import Network

if TYPE_CHECKING:
    from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome


class BaseReporter(ABC):
    """Allows for logging during the eval process"""

    def on_start_train(self, blueprint: BlueprintGenome):
        raise NotImplementedError

    def on_end_train(self, blueprint: BlueprintGenome, accuracy: float):
        raise NotImplementedError

    def on_start_epoch(self, model: Network, epoch: int):
        raise NotImplementedError

    def on_end_epoch(self, model: Network, epoch: int, loss: float, acc: float):
        raise NotImplementedError

    def on_start_batch(self, batch_idx: int, loss: float):
        raise NotImplementedError

    def on_end_batch(self, batch_idx: int, loss: float):
        raise NotImplementedError
