from __future__ import annotations

import random
from typing import List, TYPE_CHECKING, Tuple, Dict

from Genotype.NEAT.Operators.Selectors.Selector import Selector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class RouletteSelector(Selector):
    def select(self, ranked_genomes: List[int], genomes: Dict[int:Genome]) -> Tuple[Genome, Genome]:
        return tuple(random.choices(genomes.values(), weights=[genome.rank for genome in genomes.values()], k=2))
