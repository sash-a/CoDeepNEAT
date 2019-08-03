import random
import sys

import math

from src.Config.Config import Config


class Species:
    def __init__(self, representative):
        # Possible extra attribs:
        # age, hasBest, noImprovement
        self.representative = representative
        self.members = [representative]
        self.next_species_size = 1000
        self.fitness = -1
        self.age = 0

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return "Species age:" + repr(self.age) + " type: " + repr(type(self.members[0]))

    def __getitem__(self, item):
        if item >= len(self.members):
            raise Exception("index out of bounds, ased for indv:", item, "but only", len(self.members), "members")
        return self.members[item]

    def add(self, individual):
        self.members.append(individual)

    def step(self, mutation_record):
        if len(self.members) == 0:
            raise Exception("cannot step empty species")

        if self.next_species_size == 0:
            self.members = []
            return

        self._rank_species()
        elite_count = min(math.ceil(Config.elite_to_keep * len(self.members)), self.next_species_size)
        self._cull_species()

        self._reproduce(mutation_record, elite_count)

        if len(self.members) != self.next_species_size:
            raise Exception("created next generation but population size(" + repr(
                len(self.members)) + ")is wrong should be:(" + repr(self.next_species_size) + ")")

        self._select_representative()
        self.age += 1

    def _reproduce(self, mutation_record, number_of_elite):
        elite = self.members[:number_of_elite]
        children = []
        tries = 100 * (self.next_species_size - len(elite))

        while len(children) + len(elite) < self.next_species_size:
            parent1 = random.choice(self.members)
            parent2 = random.choice(self.members)

            if parent1 == parent2:
                if not parent1.validate():
                    print("invalid parent trav dict:",
                          parent1._get_traversal_dictionary(exclude_disabled_connection=True))
                    raise Exception("invalid parent in species members list", parent1)

            best = parent1 if parent1 < parent2 else parent2
            worst = parent1 if parent1 > parent2 else parent2

            child = best.crossover(worst)
            if child is None:
                raise Exception("Error: cross over produced null child")

            if child.validate():
                child = child.mutate(mutation_record)
                children.append(child)

            tries -= 1
            if tries == 0:
                raise Exception("Error: Species " + repr(self) + " failed to create enough healthy offspring. " + repr(
                    len(children)) + "/" + repr(self.next_species_size - len(elite)) + " num members=" + repr(
                    len(self.members)))

        children.extend(elite)
        if Config.maintain_module_handles:
            for i in range(number_of_elite, len(self.members)):
                member = self.members[i]
                self.members[i] = None
                del member

        self.members = children

    def _rank_species(self):
        # note original checks for equal fitnesses and chooses one with more genes
        self.members.sort(key=lambda x: x.rank)

    def get_average_rank(self):
        return sum([indv.rank for indv in self.members]) / len(self.members)

    def _cull_species(self):
        # if self.age < 3:
        #     return
        surivors = math.ceil(Config.percent_to_reproduce * len(self.members))
        if Config.maintain_module_handles:
            for i in range(surivors, len(self.members)):
                member = self.members[i]
                self.members[i] = None
                del member

        self.members = self.members[:surivors]

    def _select_representative(self):
        if not Config.speciation_overhaul:
            self.representative = random.choice(self.members)
        else:
            lowest_sum = sys.maxsize
            best_candidate = None
            for candidate in self.members:
                sum = 0
                for other in self.members:
                    if candidate == other:
                        continue
                    dist = candidate.distance_to(other)
                    if other.distance_to(candidate) != dist:
                        raise Exception("distance function not symmetrical")
                    sum += dist

                if sum < lowest_sum:
                    lowest_sum = sum
                    best_candidate = candidate
            self.representative = best_candidate

    def empty_species(self):
        self.members = []

    def set_next_species_size(self, species_size):
        self.next_species_size = species_size

    def sample_individual(self, debug=False):
        index = random.randint(0, len(self.members) - 1)
        # if debug:
        #     print("sampling random indv:",index, "length:", len(self.members))
        return self.members[index], index

    def get_species_type(self):
        if len(self.members) > 0:
            return type(self.members[0])
