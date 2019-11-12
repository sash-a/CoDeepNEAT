from abc import ABC, abstractmethod
from typing import List

from src2.Genotype.Mutagen.Mutagen import Mutagen
from src2.Genotype.NEAT.Operators.Mutators.MutationReport import MutationReport


class Gene(ABC):
    """base class for a neat node and connection"""

    def __init__(self, id: int):
        self.id: int = id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    @abstractmethod
    def get_all_mutagens(self) -> List[Mutagen]:
        raise NotImplementedError('get_all_mutagens should not be called from Gene')

    def mutate(self) -> MutationReport:
        report = MutationReport()
        for mutagen in self.get_all_mutagens():
            report += mutagen.mutate()
        return report
