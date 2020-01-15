from typing import Iterable

from src2.genotype.neat.genome import Genome
from src2.genotype.neat.operators.population_rankers.population_ranker import PopulationRanker


class TwoObjectiveRank(PopulationRanker):

    def rank(self, individuals: Iterable[Genome]) -> None:
        self.cdn_rank(individuals)

    def cdn_pareto_front(self, individuals: Iterable[Genome]):
        ordered_indvs = sorted(individuals, key=lambda indv: indv.accuracy)

        pf = [ordered_indvs[0]]  # pareto front populated with best individual in primary objective
        # each subsequent indv added must have a lower or equal primary fitness

        for indv in ordered_indvs[1:]:
            # here indv is guaranteed to have lower or equal acc to pf[0]
            # hard coded to use network sizes as second obj
            if indv.net_size < pf[0].net_size:
                # indv in question beats the last member in net size
                # inductively - indv beats all members of the pf in net size
                pf.append(indv)

        return pf

    def cdn_rank(self, individuals: Iterable[Genome]):
        ranked_individuals = []
        fronts = []
        remaining_individuals = set(individuals)

        while len(remaining_individuals) > 0:
            pf = self.cdn_pareto_front(list(remaining_individuals))
            fronts.append(pf)
            remaining_individuals = remaining_individuals - set(pf)
            ranked_individuals.extend(pf)

        num_individuals = len(ranked_individuals)
        for i, indv in enumerate(ranked_individuals):
            # i=0 is the highest rank, high rank is better
            indv.rank = num_individuals - i
        return fronts
