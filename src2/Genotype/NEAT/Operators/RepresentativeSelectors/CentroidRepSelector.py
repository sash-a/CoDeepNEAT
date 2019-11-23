from __future__ import annotations
import sys
from typing import List, TYPE_CHECKING, Dict

from src2.Genotype.NEAT.Operators.RepresentativeSelectors.RepresentativeSelector import RepresentativeSelector

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class CentroidRepSelector(RepresentativeSelector):
    """Finds the member that is most similar to all other members of the species"""

    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        lowest_sum = sys.maxsize
        best_candidate = None
        for candidate in genomes.values():
            total_distance = 0
            for other in genomes.values():
                if candidate == other:
                    continue
                dist = candidate.distance_to(other)
                # This takes extra time
                # if other.distance_to(candidate) != dist:
                #     raise Exception('distance function not symmetrical')
                total_distance += dist

            if total_distance < lowest_sum:
                lowest_sum = total_distance
                best_candidate = candidate

        return best_candidate
