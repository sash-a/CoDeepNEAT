from src.NEAT.Population import Population
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome

import src.Config.NeatProperties as Props

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
            if other_indv.distance_to(indv) <= self.speciation_thresh:
                shared_fitness += 1

        # TODO how to do this for multiobjective populations
        aggregated_fitness = sum([f * f for f in indv.fitness])  # sum of squares of all the fitness values
        indv.adjusted_fitness = aggregated_fitness / shared_fitness

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
