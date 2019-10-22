from typing import Tuple, Dict, Type, List

import math
import random

from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Operators.Mutations import MutationRecord
from src2.Configuration import config
from src2.Genotype.NEAT.Operators import Selector
from src2.Genotype.NEAT.Operators.Mutations import Mutator
import src2.Genotype.NEAT.Operators.Cross as Cross


class Species:
    species_id = 0

    def __init__(self, representative, selector, mutator):
        self.id: int = Species.species_id
        Species.species_id += 1

        self.selector: Selector = selector
        self.mutator: Mutator = mutator

        self.representative: Type[Genome] = representative
        self.members: List[Type[Genome]] = [representative.id]  # TODO dict of member_ids:members
        self.next_species_size = 1000
        self.fitness = -1
        self.tie_count = 0  # a count of how many ties there are for the top accuracy

    def sample_individual(self) -> Tuple[Type[Genome], int]:
        """:return a random individual from the species"""
        index = random.randint(0, len(self.members) - 1)
        return self.members[index], index

    def _get_num_elite(self) -> int:
        """
        Finds the number of elite this population should have, given the desired number of elite, population size and
        number of ties there are for members with the best fitness
        """
        elite = min(config.elite, len(self.members))
        highest_acc = self.members[0].fitness_values[0]
        max_fitness_ties = sum(genome.fitness_values[0] == highest_acc for genome in self.members)
        return max(elite, max_fitness_ties)

    def _unfill(self):
        """Removes all poorly performing genomes"""
        self.members = self.members[:math.ceil(len(self.members) * config.reproduce_percent)]

    def _fill(self, mutation_record: MutationRecord):
        """Fills species until it has next_species_size members"""
        children: List[Genome] = []
        elite = self._get_num_elite()
        self.selector.before_selection(self.members)

        while len(children) < self.next_species_size - elite:
            p1, p2 = self.selector.select(self.members)
            child = Cross.over(p1, p2)
            self.mutator.mutate(child, mutation_record)
            children.append(child)

        self.members = self.members[elite] + children

    def _select_representative(self):
        # TODO
        #  Possibly pass in rep selection function so can do different types of rep selection
        pass

    def step(self, mutation_record: MutationRecord):
        if len(self.members) == 0:
            raise Exception('Cannot step empty species')

        if self.next_species_size == 0:
            self.members = []
            return

        # note original CoDeepNEAT checks for equal fitness's and prioritizes genomes with more genes
        self.members.sort(key=lambda genome: genome.rank)
        self._unfill()
        self._fill(mutation_record)

        self._select_representative()
