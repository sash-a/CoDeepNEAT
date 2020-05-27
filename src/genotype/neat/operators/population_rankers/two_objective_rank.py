from __future__ import annotations

import copy
from typing import Iterable, Dict, TYPE_CHECKING, Optional, Tuple, List

from configuration import config
from src.genotype.neat.operators.population_rankers.population_ranker import PopulationRanker
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class TwoObjectiveRank(PopulationRanker):

    def __init__(self):
        super().__init__(2)

    def rank(self, individuals: Iterable[Genome], value_coefficients: List[int] = None) -> None:
        num_indvs = len(list(copy.deepcopy(individuals)))
        fronts = self.cdn_rank(individuals)
        if config.visualise_moo_scores:
            self.plot_scores(fronts, num_indvs)

    @staticmethod
    def cdn_pareto_front(individuals: Iterable[Genome]):
        acc_ordered_indvs = sorted(individuals, key=lambda indv: indv.aggregated_acc, reverse=True)
        # print("accs:",[indv.accuracy for indv in acc_ordered_indvs])

        pf = [acc_ordered_indvs[0]]  # pareto front populated with best individual in primary objective
        # each subsequent indv added must have a lower or equal primary fitness

        for indv in acc_ordered_indvs[1:]:
            # here indv is guaranteed to have lower or equal acc to pf[0]
            # hard coded to use network sizes as second obj
            if indv.net_size < pf[-1].net_size:
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
        print("num ranked indvs", num_individuals)
        for i, indv in enumerate(ranked_individuals):
            # rank=0 is the least fit
            indv.rank = num_individuals - i
        return fronts

    def plot_scores(self, fronts, num_individuals):
        for front_no in range(len(fronts)):
            label = "front:" + str(front_no)
            xxs = []
            yys = []

            for indv in fronts[front_no]:
                colour = self.get_rank_colour(indv, num_individuals)
                xxs.append(indv.aggregated_acc)
                yys.append(indv.net_size)

            # plt.scatter(xxs, yys, color=colour)
            plt.scatter(xxs, yys, color=colour, label=label)

        plt.legend()
        plt.show()

    @staticmethod
    def get_rank_colour(individual, num_individuals):
        skew_fac = 0.55  # 0.55

        frac = lambda indv: indv.rank / num_individuals
        diff = lambda indv: 2 * (frac(indv) - 0.5)  # from -1 to 1, diff< 0 means shrink frac
        dist = lambda indv: abs(diff(indv))  # 0 : 1   ,  0 being frac = 0.5

        adjusted_frac = lambda indv: pow(frac(indv), skew_fac ** dist(indv)) if frac(indv) < 0.5 else pow(frac(indv), (
                1 / skew_fac) ** dist(indv))

        rank_colouring = lambda indv: (1 - adjusted_frac(indv), adjusted_frac(indv), 0.5 * adjusted_frac(indv) ** 2)
        return rank_colouring(individual)
