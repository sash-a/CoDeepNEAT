from __future__ import annotations

import random
from typing import List, TYPE_CHECKING, Tuple, Dict

from src2.Genotype.NEAT.Operators.ParentSelectors.Selector import Selector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class UniformSelector(Selector):
    def select(self, ranked_genomes: List[int], genomes: Dict[int:Genome]) -> Tuple[Genome, Genome]:
        return tuple(random.choices(list(genomes.values()), k=2))
