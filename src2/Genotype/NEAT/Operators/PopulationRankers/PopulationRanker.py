from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from Genotype.NEAT.Genome import Genome


class PopulationRanker(ABC):
    @abstractmethod
    def rank(self, individuals: Iterable[Genome]) -> None:
        pass
