from typing import Iterable

from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Operators.PopulationRankers.PopulationRanker import PopulationRanker


class SingleObjectiveRank(PopulationRanker):
    def rank(self, individuals: Iterable[Genome]) -> None:
        """Sorts individuals and sets their ranks. Where 1 is the lowest and len(individuals) is the highest rank"""
        ordered_indvs = sorted(individuals, key=lambda indv: indv.accuracy)
        for i, indv in enumerate(ordered_indvs, 1):
            indv.rank = i
