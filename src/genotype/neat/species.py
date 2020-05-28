from __future__ import annotations

import math
import random
from typing import List, Dict, TYPE_CHECKING

import src.genotype.neat.operators.cross as Cross
from configuration import config
from src.genotype.neat import mutation_record
from src.genotype.neat.operators.mutators import mutator
from src.genotype.neat.operators.parent_selectors import selector
from src.genotype.neat.operators.representative_selectors.representative_selector import RepresentativeSelector

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class Species:
    species_id_counter = 0

    # These are set in the populations init method
    selector: selector
    representative_selector: RepresentativeSelector

    def __init__(self, representative: Genome, mutator: mutator):
        self.id: int = Species.species_id_counter
        Species.species_id_counter += 1

        self.mutator: mutator = mutator

        self.representative: Genome = representative
        self.members: Dict[int, Genome] = {representative.id: representative}
        self.ranked_members: List[int] = [representative.id]
        self.next_species_size: int = 1000
        self.max_fitness_ties: int = 0  # a count of how many ties there are for the top accuracy

    def __iter__(self):
        return iter(self.members.values())

    def __len__(self):
        return len(self.members)

    def __getitem__(self, id: int):
        return self.members[id]

    def __repr__(self):
        if len(self.members) == 0:
            return "Species empty"
        return 'Species has ' + repr(len(self.members)) + ' members of type: ' + repr(type(list(self)[0]))

    def add(self, individual: Genome):
        self.members[individual.id] = individual

    def sample_individual(self) -> Genome:
        """:return a random individual from the species"""
        if len(self.members.values()) == 0:
            raise Exception("cannot sample from empty species")

        member = random.choice(list(self.members.values()))
        if member is None:
            raise Exception("none member in species")
        return member

    def _get_num_elite(self) -> int:
        """
        Finds the number of elite this population should have, given the desired number of elite, population size and
        number of ties there are for members with the best fitness
        """
        elite = math.ceil(config.elite_percent * self.next_species_size)
        if elite == 0:  # don't want to count ties if there are no elite anyways
            return elite

        highest_acc = self.members[self.ranked_members[0]].aggregated_acc
        self.max_fitness_ties = \
            sum(genome.aggregated_acc == highest_acc for genome in self.members.values())  # TODO test
        return max(elite, self.max_fitness_ties)

    def _unfill(self):
        """Removes all poorly performing genomes"""
        self.ranked_members = self.ranked_members[:math.ceil(len(self.ranked_members) * config.reproduce_percent)]
        survivors = set(self.ranked_members)
        new_members = {}
        for id in self.members.keys():
            if id in survivors:
                new_members[id] = self.members[id]

        self.members = new_members

    def _fill(self, mutation_record: mutation_record):
        """Fills species until it has next_species_size members, using crossover and mutation"""
        children: List[Genome] = []
        num_elite = self._get_num_elite()
        while len(children) < self.next_species_size - num_elite:
            p1, p2 = Species.selector.select(ranked_genomes=self.ranked_members, genomes=self.members)
            child = Cross.over(p1, p2)
            self.mutator.mutate(child, mutation_record)
            if child.validate():
                children.append(child)

        self.members = {id: self.members[id] for id in self.ranked_members[:num_elite]}  # adds in elite
        self.members = {**self.members, **{child.id: child for child in children}}  # adds in children

    def step(self, mutation_record: mutation_record):
        """Runs a single generation of evolution"""
        if not self.members:
            raise Exception('Cannot step empty species')

        if self.next_species_size == 0:
            # this species has been extincted
            self.members = {}
            self.ranked_members = []
            return

        # note original CoDeepNEAT checks for equal fitness's and prioritizes genomes with more genes
        self.ranked_members = [id for id in self.members.keys()]
        self.ranked_members.sort(key=lambda id: self.members[id].rank, reverse=True)  # Higher rank = better
        self._unfill()
        self._fill(mutation_record)

        self.representative = Species.representative_selector.select_representative(self.members)
        # print('done step')

    def clear(self):
        self.members: Dict[int, Genome] = {}
        self.ranked_members: List[int] = []
