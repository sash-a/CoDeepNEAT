from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class RepresentativeSelector(ABC):
    """Finds a representative for a species"""

    @abstractmethod
    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        pass
