from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class Selector:
    def __init__(self):
        pass

    def select(self, genomes: List[Genome]) -> Tuple[Genome, Genome]:
        pass
