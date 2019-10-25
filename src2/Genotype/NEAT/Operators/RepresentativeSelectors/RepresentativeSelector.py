from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from Genotype.NEAT.Genome import Genome


class RepresentativeSelector(ABC):
    """Finds a representative for a species"""

    @abstractmethod
    def select_representative(self, members: List[Genome]) -> Genome:
        pass
