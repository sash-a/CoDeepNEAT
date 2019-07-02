from src.NEAT.Population import Population
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome

import src.NEAT.NeatProperties as Props

from typing import List, Union


class MultiobjectivePopulation(Population):
    def __init__(self, population: List[Union[BlueprintGenome, ModuleGenome]], mutations: dict):
        super().__init__(population, mutations)

    def save_elite(self, species):
        pf = species.pareto_front()
        species.members = species.members[len(pf):]

        while species.members:
            next_front = species.pareto_front()
            pf.extend(next_front)
            species.members = species.members[len(next_front):]

        # editing the old species
        species.representative = pf[0]

        members_to_save = Props.PERCENT_TO_SAVE * len(pf)
        pf = pf[:members_to_save]

        # TODO CDN say species.members=pf, but this doesn't make sense given how Population works
        # species.members = pf
        species.members.clear()

        return pf
