import random
import sys
from abc import ABC, abstractmethod
from typing import List

from Genotype.NEAT.Genome import Genome


class RepresentativeSelector(ABC):
    """Finds a representative for a species"""

    @abstractmethod
    def select_representative(self, members: List[Genome]) -> Genome:
        pass


class RandomRepSelector(RepresentativeSelector):
    def select_representative(self, members: List[Genome]) -> Genome:
        return random.choice(members)


class MostSimilarRepSelector(RepresentativeSelector):
    """Finds the member that is most similar to all other members of the species"""

    def select_representative(self, members: List[Genome]) -> Genome:
        lowest_sum = sys.maxsize
        best_candidate = None
        for candidate in members:
            total_distance = 0
            for other in members:
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


class BestRepresentativeSelection(RepresentativeSelector):
    def select_representative(self, members: List[Genome]) -> Genome:
        return max(members, key=lambda genome: genome.fitness_values[0])
