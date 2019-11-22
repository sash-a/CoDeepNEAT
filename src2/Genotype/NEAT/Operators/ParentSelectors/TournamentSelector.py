from __future__ import annotations

import random
from typing import List, TYPE_CHECKING, Tuple, Dict

from src2.Genotype.NEAT.Operators.ParentSelectors.Selector import Selector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class TournamentSelector(Selector):
    def __init__(self, k):
        self.k = k

    def select(self, ranked_genomes: List[int], genomes: Dict[int:Genome]) -> Tuple[Genome, Genome]:
        tournament = random.choices(list(genomes.values()), k=self.k)
        parent1 = max(tournament, key=lambda genome: genome.rank)

        tournament = random.choices(list(genomes.values()), k=self.k)
        parent2 = max(tournament, key=lambda genome: genome.rank)
        return parent1, parent2
