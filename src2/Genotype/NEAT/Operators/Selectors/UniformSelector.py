from __future__ import annotations

import random
from typing import List, TYPE_CHECKING, Tuple

from Genotype.NEAT.Operators.Selectors.Selector import Selector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class UniformSelector(Selector):
    def select(self, genomes: List[Genome]) -> Tuple[Genome, Genome]:
        return tuple(random.choices(genomes, k=2))
