from __future__ import annotations

import random
from typing import List, TYPE_CHECKING, Tuple, Dict

from src.genotype.neat.operators.parent_selectors.selector import Selector

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class RouletteSelector(Selector):
    def select(self, ranked_genomes: List[int], genomes: Dict[int:Genome]) -> Tuple[Genome, Genome]:
        return tuple(random.choices(list(genomes.values()), weights=[genome.rank for genome in genomes.values()], k=2))
