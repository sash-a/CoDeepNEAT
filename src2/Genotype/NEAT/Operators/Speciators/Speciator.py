from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from Genotype.NEAT.Species import Species


class Speciator(ABC):
    def __init__(self, threshold: float, target_num_species: int):
        self.threshold: float = threshold
        self.target_num_species: int = target_num_species
        self.current_threshold_dir: int = 1

    @abstractmethod
    def speciate(self, species: List[Species]) -> None:
        pass

    @abstractmethod
    def adjust_speciation_threshold(self, n_species: int) -> None:
        """
        dynamically alters the speciation threshold to try to achieve the target number of species for the next
        speciation step
        """
        pass
