from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Species import Species
    from src2.Genotype.NEAT.Operators.Mutators.Mutator import Mutator


class Speciator(ABC):
    def __init__(self, threshold: float, target_num_species: int,
                 mutator: Mutator):
        self.threshold: float = threshold
        self.target_num_species: int = target_num_species
        self.current_threshold_dir: int = 1
        self.mutator: Mutator = mutator

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
