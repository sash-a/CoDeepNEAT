from typing import List, Union
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome
from src.CoDeepNEAT.ModuleGenome import ModuleGenome


def CDN_pareto(individuals: List[Union[BlueprintGenome, ModuleGenome]]):
    # Sorting members on primary fitness
    individuals.sort(key=lambda indv: indv.fitness[0])
    pf = [individuals[0]]  # pareto front populated with best individual in primary objective

    for indv in individuals[1:]:
        if indv.fitness[1] > pf[-1].fitness[1]:
            pf.append(indv)

    return pf
