from typing import Iterable

from Genotype.NEAT.Genome import Genome
from Genotype.NEAT.Operators.PopulationRankers.PopulationRanker import PopulationRanker


class SingleObjectiveRank(PopulationRanker):
    def rank(self, individuals: Iterable[Genome]) -> None:
        """Sorts individuals and sets their ranks. Where 0 is the lowest and len(individuals) is the highest rank"""
        ordered_indvs = sorted(individuals, key=lambda indv: indv.fitness_values[0], reverse=True)
        for i, indv in enumerate(ordered_indvs):
            indv.rank = i
