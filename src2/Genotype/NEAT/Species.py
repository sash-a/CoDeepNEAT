from typing import Tuple, List

import math
import random

from Genotype.NEAT.Operators.Crowders.Crowder import Crowder
from Genotype.NEAT.Operators.RepresentativeSelectors.RepresentativeSelector import RepresentativeSelector
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Operators.Mutations import MutationRecord
from src2.Configuration import config
from Genotype.NEAT.Operators.Selectors import Selector
from src2.Genotype.NEAT.Operators.Mutations import Mutator
import src2.Genotype.NEAT.Operators.Cross as Cross


class Species:
    species_id = 0

    # These are set in the populations init method
    selector: Selector
    mutator: Mutator
    representative_selector: RepresentativeSelector
    crowder: Crowder

    def __init__(self, representative: Genome):
        self.id: int = Species.species_id
        Species.species_id += 1

        self.representative: Genome = representative
        self.members: List[Genome] = [representative.id]  # TODO dict of member_ids:members
        self.next_species_size: int = 1000
        self.fitness: int = -1
        self.max_fitness_ties: int = 0  # a count of how many ties there are for the top accuracy

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item: int):
        if item >= len(self.members):
            raise IndexError('Index out of bounds: ' + str(item) + ' max: ' + str(len(self.members)))
        return self.members[item]

    def __repr__(self):
        return 'Species has ' + repr(len(self.members)) + ' members of type: ' + repr(type(self.members[0]))

    def add(self, individual: Genome):
        self.members.append(individual)

    def sample_individual(self) -> Tuple[Genome, int]:
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
        self.max_fitness_ties = sum(genome.fitness_values[0] == highest_acc for genome in self.members)  # TODO test
        return max(elite, self.max_fitness_ties)

    def _unfill(self):
        """Removes all poorly performing genomes"""
        self.members = self.members[:math.ceil(len(self.members) * config.reproduce_percent)]

    def _fill(self, mutation_record: MutationRecord):
        """Fills species until it has next_species_size members, using crossover and mutation"""
        children: List[Genome] = []
        elite = self._get_num_elite()
        Species.selector.before_selection(self.members)

        while len(children) < self.next_species_size - elite:
            p1, p2 = Species.selector.select(self.members)
            child = Cross.over(p1, p2)
            Species.mutator.mutate(child, mutation_record)
            children.append(child)

        # Only use the parent population after the elite, because elite will be added regardless
        next_generation_members = Species.crowder.crowd(children, self.members[:elite])
        self.members = self.members[:elite] + next_generation_members

    def step(self, mutation_record: MutationRecord):
        """Runs a single generation of evolution"""
        if not self.members:
            raise Exception('Cannot step empty species')

        if self.next_species_size == 0:
            self.members = []
            return

        # note original CoDeepNEAT checks for equal fitness's and prioritizes genomes with more genes
        self.members.sort(key=lambda genome: genome.rank)  # TODO might need to reverse depends how we set rank
        self._unfill()
        self._fill(mutation_record)

        self.representative = Species.representative_selector.select_representative(self.members)
