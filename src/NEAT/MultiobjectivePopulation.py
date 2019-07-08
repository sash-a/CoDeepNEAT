from src.NEAT.Population import Population
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome

import src.NEAT.NeatProperties as Props

from typing import List, Union


class MultiobjectivePopulation(Population):
    def __init__(self, population: List[Union[BlueprintGenome, ModuleGenome]], mutations: dict,
                 pop_size: int, node_mutation_chance: float, connection_mutation_chance: float,
                 target_num_species: int):
        super().__init__(population, mutations, pop_size, node_mutation_chance, connection_mutation_chance,
                         target_num_species)

    def adjust_fitness(self, indv: Union[BlueprintGenome, ModuleGenome]):
        shared_fitness = 0
        for other_indv in self.individuals:
            # TODO should you measure distance to self?
            # Result will always be 1, but allows for species of a single member
            # if other_indv == indv:
            #     continue

            if other_indv.distance_to(indv) <= Props.SPECIES_DISTANCE_THRESH:
                shared_fitness += 1

        # TODO how to do this for multiobjective populations
        aggeraged_fitness = sum([f * f for f in indv.fitness])
        indv.adjusted_fitness = aggeraged_fitness / shared_fitness

    def save_elite(self, species):
        pf = species.pareto_front()
        species.members = species.members[len(pf):]

        while species.members:
            next_front = species.pareto_front()
            pf.extend(next_front)
            species.members = species.members[len(next_front):]

        members_to_save = max(2, int(Props.PERCENT_TO_SAVE * len(pf)))
        pf = pf[:members_to_save]
        species.members = pf

        return pf
