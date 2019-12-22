from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Tuple, Dict

if TYPE_CHECKING:
    from src2.genotype.neat.genome import Genome


class Selector(ABC):
    @abstractmethod
    def select(self, ranked_genomes: List[int], genomes: Dict[int:Genome]) -> Tuple[Genome, Genome]:
        pass
