from __future__ import annotations

from typing import Iterable, List, TYPE_CHECKING

from src.genotype.neat.operators.population_rankers.population_ranker import PopulationRanker

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class SingleObjectiveRank(PopulationRanker):

    def __init__(self):
        super().__init__(1)

    def rank(self, individuals: Iterable[Genome], value_coefficients:List[int] = None) -> None:
        """Sorts individuals and sets their ranks. Where 1 is the lowest and len(individuals) is the highest rank"""
        ordered_indvs = sorted(individuals, key=lambda indv: indv.aggregated_acc)
        for i, indv in enumerate(ordered_indvs, 1):
            indv.rank = i
