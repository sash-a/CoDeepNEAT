from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class PopulationRanker(ABC):
    num_objectives: int

    def __init__(self, num_objectives : int):
        self.num_objectives = num_objectives

    @abstractmethod
    def rank(self, individuals: Iterable[Genome]) -> None:
        """Ranks individuals based on their fitness, with the rank 0 being the worst rank"""
