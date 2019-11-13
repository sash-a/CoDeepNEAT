from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict

from src2.Genotype.NEAT.Operators.RepresentativeSelectors.RepresentativeSelector import RepresentativeSelector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class RandomRepSelector(RepresentativeSelector):
    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        return random.choice(list(genomes.values()))
