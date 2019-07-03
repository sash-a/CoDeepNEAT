from src.NEAT.Genome import Genome
from src.NEAT.Species import Species
from src.NEAT.Crossover import crossover
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
import src.NEAT.NeatProperties as Props

import random
from typing import List, Union

"""
Population persists across whole run time
"""


class Population:

    def __init__(self, population: List[Union[BlueprintGenome, ModuleGenome, Genome]], mutations: dict):
        """
        :param population: list of all individuals
        """

        self.mutations = mutations
        self.curr_innov = max(indv.connections[-1].innovation for indv in population)
        self.max_node_id = 6  # TODO len(population)  # this assumes that no nodes are disabled in initial population

        # Either connection mutation: tuple(nodeid,nodeid) : innovation number
        # Or node mutation: innovation number : nodeid
        self.mutation = dict()

        self.individuals: List[Union[BlueprintGenome, ModuleGenome, Genome]] = population

        self.speciation_thresh = Props.SPECIES_DISTANCE_THRESH
        self.species: List[Species] = []
        self.speciate(True)

    def speciate(self, first_gen=False):
        """
        Place all individuals in their first compatible species if one exists
        Otherwise create new species with current individual as its representative

        :return: list of species
        """
        # Remove all current members from the species
        for spc in self.species:
            spc.members.clear()

        # Placing individuals in their correct species
        for individual in self.individuals:
            found_species = False
            for spc in self.species:
                if spc.is_compatible(individual, thresh=self.speciation_thresh):
                    spc.add_member_safe(individual)
                    found_species = True
                    break

            if not found_species:
                self.species.append(Species(individual))

        # Remove all empty species
        self.species = [spc for spc in self.species if spc.members]

        # Dynamic speciation threshold
        if not first_gen:
            mod = 1
            if len(self.species) < Props.TARGET_NUM_SPECIES:
                mod = -1

            self.speciation_thresh = max(0.3, self.speciation_thresh + (mod * Props.SPECIES_DISTANCE_THRESH_MOD))

        return self.species

    def adjust_fitness(self, indv: Genome):
        shared_fitness = 0
        for other_indv in self.individuals:
            # TODO should you measure distance to self?
            # Result will always be 1, but allows for species of a single member
            # if other_indv == indv:
            #     continue

            if other_indv.distance_to(indv) <= Props.SPECIES_DISTANCE_THRESH:
                shared_fitness += 1

        indv.adjusted_fitness = indv.fitness / shared_fitness

    def save_elite(self, species):
        # only allow top x% to reproduce
        # return species sorted by fitness
        species.members.sort(key=lambda indv: indv.fitness, reverse=True)
        # min of two because need two parents to crossover
        num_remaining_mem = max(2, int(len(species.members) * Props.PERCENT_TO_SAVE))
        remaining_members = species.members[:num_remaining_mem]

        return remaining_members

    def step(self):
        new_pop = []

        # calculate adjusted fitness
        tot_adj_fitness = 0
        for indv in self.individuals:
            self.adjust_fitness(indv)
            tot_adj_fitness += indv.adjusted_fitness

        # Reproduce within species
        for spc in self.species:
            # find num_children given adjusted fitness sum for species
            species_adj_fitness = sum([x.adjusted_fitness for x in spc.members])
            num_children = max(Props.MIN_CHILDREN_PER_SPECIES, int((species_adj_fitness / tot_adj_fitness) * (
                    Props.POP_SIZE - Props.PERCENT_TO_SAVE * Props.POP_SIZE)))

            remaining_members = self.save_elite(spc)
            # Add elite back into new population
            new_pop.extend(remaining_members)

            # Create children
            if remaining_members:
                for _ in range(num_children):
                    parent1 = random.choice(remaining_members)
                    parent2 = random.choice(remaining_members)

                    child = crossover(parent1, parent2)

                    self.curr_innov, self.max_node_id = child.mutate(self.mutations,
                                                                     self.curr_innov,
                                                                     self.max_node_id,
                                                                     Props.NODE_MUTATION_CHANCE,
                                                                     Props.CONNECTION_MUTATION_CHANCE)

                    new_pop.append(child)

        self.individuals = new_pop
        self.speciate()
