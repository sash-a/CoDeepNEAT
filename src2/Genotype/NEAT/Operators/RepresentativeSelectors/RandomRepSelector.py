import random
from typing import List, TYPE_CHECKING, Dict

from src2.Genotype.NEAT.Operators.RepresentativeSelectors.RepresentativeSelector import RepresentativeSelector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class RandomRepSelector(RepresentativeSelector):
    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        return random.choice(genomes.values())
