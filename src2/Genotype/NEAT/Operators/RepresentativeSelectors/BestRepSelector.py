from __future__ import annotations
from typing import List, TYPE_CHECKING, Dict

from src2.Genotype.NEAT.Operators.RepresentativeSelectors.RepresentativeSelector import RepresentativeSelector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class BestRepSelector(RepresentativeSelector):
    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        return max(genomes.values(), key=lambda genome: genome.accuracy)
