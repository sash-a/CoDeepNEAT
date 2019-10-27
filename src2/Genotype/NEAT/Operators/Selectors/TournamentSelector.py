from __future__ import annotations

import random
from typing import List, TYPE_CHECKING, Tuple

from Genotype.NEAT.Operators.Selectors.Selector import Selector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class TournamentSelector(Selector):
    def __init__(self, k):
        self.k = k

    def select(self, genomes: List[Genome]) -> Tuple[Genome, Genome]:
        tournament = random.choices(genomes, k=self.k)
        parent1 = max(tournament, key=lambda genome: genome.rank)
        parent2 = max(tournament, key=lambda genome: genome.rank)
        return parent1, parent2
