from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class PopulationRanker(ABC):
    num_objectives: int

    def __init__(self, num_objectives : int):
        self.num_objectives = num_objectives

    @abstractmethod
    def rank(self, individuals: Iterable[Genome], value_coefficients:List[int] = None) -> None:
        """
            value_coefficients, if given are multiplied into objective scores
            the nth element is multiplied into all obj_n scores

            Ranks individuals based on their fitness, with the rank 0 being the worst rank
            Rankers try to maximise the objective scores along each objective dimension
            If an objective requires minimisation such as network size - the ranker should be informed via the
            If informed of a minimising obj, the ranker will maximise the negative values
        """
