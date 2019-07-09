from src.NEAT.Genome import Genome
from src.NEAT.Species import Species
from src.NEAT.Crossover import crossover
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
import src.Config.NeatProperties as Props

import random
from typing import List, Union

"""
Population persists across whole run time
"""


class Population:

    def __init__(self, population: List[Union[BlueprintGenome, ModuleGenome, Genome]], mutations: dict,
                 ideal_pop_size: int, node_mutation_chance: float, connection_mutation_chance: float,
                 target_num_species: int):
        """
        :param population: list of all individuals
        """

        self.mutations = mutations
        self.curr_innov = max(indv.connections[-1].innovation for indv in population)
        self.max_node_id = max(indv.nodes[-1].id for indv in population)

        # Either connection mutation: tuple(nodeid,nodeid) : innovation number
        # Or node mutation: innovation number : nodeid
        self.mutation = dict()

        self.individuals: List[Union[BlueprintGenome, ModuleGenome, Genome]] = population

        # Speciation
        self.target_num_species = target_num_species
        self.num_species_mod_dir = 0
        self.num_species_mod = Props.SPECIES_DISTANCE_THRESH_MOD
        self.speciation_thresh = Props.SPECIES_DISTANCE_THRESH
        self.species: List[Species] = []
        self.speciate(True)

        self.ideal_pop_size = ideal_pop_size
        self.node_mutation_chance = node_mutation_chance
        self.connection_mutation_chance = connection_mutation_chance

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
                    spc.add_member(individual, thresh=self.speciation_thresh, safe=True)
                    found_species = True
                    break

            if not found_species:
                self.species.append(Species(individual))

        # Remove all empty species
        self.species = [spc for spc in self.species if spc.members]

        if not first_gen:
            self.dynamic_speciation()

        return self.species

    def dynamic_speciation(self):
        # Dynamic speciation threshold
        if len(self.species) < self.target_num_species:
            new_dir = -1
        elif len(self.species) > self.target_num_species:
            new_dir = 1
        else:
            new_dir = 0

        # Exponential growth
        if new_dir != self.num_species_mod_dir:
            self.num_species_mod = Props.SPECIES_DISTANCE_THRESH_MOD
        else:
            self.num_species_mod *= 2

        self.num_species_mod_dir = new_dir

        self.speciation_thresh = max(0.01, self.speciation_thresh + (self.num_species_mod_dir * self.num_species_mod))

    def adjust_fitness(self, indv: Genome):
        shared_fitness = 0
        for other_indv in self.individuals:
            if other_indv.distance_to(indv) <= self.speciation_thresh:
                shared_fitness += 1

        indv.adjusted_fitness = indv.fitness[0] / shared_fitness  # single obj therefore always fitness[0]

    def save_elite(self, species):
        # only allow top x% to reproduce
        # return species sorted by fitness
        species.members.sort(key=lambda indv: indv.fitness[0], reverse=True)
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
            # TODO this is not creating the correct number of children
            num_children = max(Props.MIN_CHILDREN_PER_SPECIES, int((species_adj_fitness / tot_adj_fitness) * (
                    self.ideal_pop_size - Props.PERCENT_TO_SAVE * self.ideal_pop_size)))

            # Ignoring defective members
            spc.members = [mem for mem in spc.members if not mem.defective]

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
                                                                     self.node_mutation_chance,
                                                                     self.connection_mutation_chance)

                    new_pop.append(child)

        self.individuals = new_pop
        self.speciate()

    def __len__(self):
        return len(self.individuals)

    def get_num_species(self):
        return len(self.species)
