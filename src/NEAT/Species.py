import random
import math
import sys

from src.Config import Config, NeatProperties as Props
from src.CoDeepNEAT.CDNGenomes import BlueprintGenome


class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]
        self.next_species_size = 1000
        self.fitness = -1
        self.age = 0
        self.tie_count = 0  # a count of how many ties there are for the top accuracy

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

    def step(self, mutation_record, topological_mutation_modifier=1, attribute_mutation_modifier=1, module_pop=None,
             gen=-1):
        if len(self.members) == 0:
            raise Exception("cannot step empty species")

        if self.next_species_size == 0:
            self.members = []
            return

        self._rank_species()
        elite_count = self.find_num_elite()

        self._cull_species(elite_count)
        self._reproduce(mutation_record, elite_count, topological_mutation_modifier, attribute_mutation_modifier,
                        module_pop, gen=gen)

        print("mutation modufiers~ top:", topological_mutation_modifier, "att:", attribute_mutation_modifier, "spc:",
              (1 / math.pow(self.fitness / 1.1, 0.9)), "fitness:", self.fitness)

        self._select_representative()
        self.age += 1

    def find_num_elite(self):
        """if the top n members have the same acc - they all survive"""
        num_elite = min(math.ceil(Props.ELITE_TO_KEEP * len(self.members)), self.next_species_size)
        highest_acc = self.members[0].fitness_values[0]
        i = 1
        # print("searching for ties from:",[self.members[i].fitness_values[0] for i in range(len(self.members))])
        while i < len(self.members) and self.members[i].fitness_values[0] == highest_acc:
            i += 1
        self.tie_count = i
        if self.members[0].fitness_values[0] == self.members[-1].fitness_values[0]:
            self.tie_count = len(self.members)
        if self.tie_count > 1:
            print("ties:", [self.members[i].fitness_values[0] for i in range(len(self.members)) if
                            self.members[i].fitness_values[0] == self.members[0].fitness_values[0]])
        return max(self.tie_count, num_elite)

    def _reproduce(self, mutation_record, number_of_elite, topological_mutation_modifier, attribute_mutation_modifier,
                   module_pop=None, gen=-1):
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
                if Config.adjust_species_mutation_magnitude_based_on_fitness:
                    """less fit species change more rapidly"""
                    attribute_mutation_magnitude = max(1 / math.pow(self.fitness, 1.2), 3)
                else:
                    attribute_mutation_magnitude = 1

                if Config.use_representative and type(child) == BlueprintGenome and module_pop is None:
                    raise Exception(
                        'Using representative, but received a none module population when mutating a blueprint node')

                child = child.mutate(mutation_record,
                                     attribute_magnitude=attribute_mutation_magnitude * attribute_mutation_modifier,
                                     topological_magnitude=topological_mutation_modifier, module_population=module_pop,
                                     gen=gen)
                children.append(child)

            tries -= 1
            if tries == 0:
                raise Exception("Error: Species " + repr(self) + " failed to create enough healthy offspring. " + repr(
                    len(children)) + "/" + repr(self.next_species_size - len(elite)) + " num members=" + repr(
                    len(self.members)))

        children.extend(elite)
        for i in range(number_of_elite, len(self.members)):
            member = self.members[i]
            self.members[i] = None
            del member

        self.members = children

    def _rank_species(self):
        # note original checks for equal fitnesses and chooses one with more genes
        self.members.sort(key=lambda x: x.rank)

    def get_average_rank(self):
        if Config.adjust_species_mutation_magnitude_based_on_fitness:
            ranks = [indv.rank for indv in self.members if indv.fitness_values[0] != 0]
            return sum(ranks) / len(ranks)
        else:
            return sum([indv.rank for indv in self.members]) / len(self.members)

    def _cull_species(self, num_elite):
        surivors = math.ceil(Props.PERCENT_TO_REPRODUCE * len(self.members))
        surivors = max(surivors, num_elite)
        for i in range(surivors, len(self.members)):
            member = self.members[i]
            self.members[i] = None
            del member  # TODO does this change the array length

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
