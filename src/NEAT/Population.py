from src.NEAT.Genotype import Genome
from src.NEAT.Species import Species
from src.NEAT.Crossover import crossover
import src.NEAT.NeatProperties as Props

import random
from typing import List

"""
Population persists across whole run time
"""


class Population:

    def __init__(self, population: List[Genome]):
        """
        :param population: list of all individuals
        """

        self.gen_mutations = set()
        self.curr_innov = max(indv.connections[-1].innovation for indv in population)
        self.max_node_id = len(population)  # this assumes that no nodes are disabled in initial population

        self.individuals = population
        self.species: List[Species] = []

        self.speciate()

    def speciate(self):
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
                if spc.is_compatible(individual):
                    spc.add_member(individual)
                    found_species = True
                    break

            if not found_species:
                self.species.append(Species(individual))

        # Remove all empty species
        species = [spc for spc in self.species if spc.members]

        return species

    def adjust_fitness(self, indv: Genome):
        shared_fitness = 0
        for other_indv in self.individuals:
            # TODO should you measure distance to self?
            # Result will always be 1, but allows for species of a single member
            # if other_indv == indv:
            #     continue

            if other_indv.distance_to(indv) <= Props.DISTANCE_THRESH:
                shared_fitness += 1

        indv.adjusted_fitness = indv.fitness / shared_fitness

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
            num_children = max(Props.MIN_CHILDREN_PER_SPECIES, int(species_adj_fitness / tot_adj_fitness) * (
                    Props.POP_SIZE - Props.PERCENT_TO_SAVE * Props.POP_SIZE))

            # only allow top x% to reproduce
            spc.members.sort(key=lambda indv: indv.fitness, reverse=True)
            # min of two bc need two parents to create child
            num_remaining_mem = max(2, int(len(spc.members) * Props.PERCENT_TO_SAVE))
            remaining_members = spc.members[:num_remaining_mem]
            spc.members.clear()  # reset species

            # Elitism
            elite = min(Props.ELITE_TO_KEEP, num_remaining_mem)
            new_pop.extend(remaining_members[:elite])

            # Create children
            for _ in range(num_children):
                parent1 = random.choice(remaining_members)
                parent2 = random.choice(remaining_members)

                child = crossover(parent1, parent2)
                child.mutate(self.gen_mutations,
                             self.curr_innov,
                             self.max_node_id,
                             Props.NODE_MUTATION_CHANCE,
                             Props.CONNECTION_MUTATION_CHANCE)

                new_pop.append(child)

        self.individuals = new_pop
        self.speciate()
        self.gen_mutations.clear()
