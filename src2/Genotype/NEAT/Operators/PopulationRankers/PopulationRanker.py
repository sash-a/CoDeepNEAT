from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class PopulationRanker(ABC):
    @abstractmethod
    def rank(self, individuals: Iterable[Genome]) -> None:
        """Ranks individuals based on their fitness, with the rank 0 being the worst rank"""
        pass
