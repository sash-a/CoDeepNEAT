import random
from typing import List, TYPE_CHECKING, Dict

from Genotype.NEAT.Operators.RepresentativeSelectors.RepresentativeSelector import RepresentativeSelector

if TYPE_CHECKING:
    from Genotype.NEAT.Genome import Genome


class RandomRepSelector(RepresentativeSelector):
    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        return random.choice(genomes.values())
