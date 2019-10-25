from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from Genotype.NEAT.Genome import Genome


class Crowder(ABC):
    @abstractmethod
    def crowd(self, children: List[Genome], parents: List[Genome]) -> List[Genome]:
        pass
