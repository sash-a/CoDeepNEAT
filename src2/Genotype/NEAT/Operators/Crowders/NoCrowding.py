from typing import List, TYPE_CHECKING

from Genotype.NEAT.Operators.Crowders.Crowder import Crowder

if TYPE_CHECKING:
    from Genotype.NEAT.Genome import Genome


class NoCrowding(Crowder):
    def crowd(self, children: List[Genome], parents: List[Genome]) -> List[Genome]:
        return children
