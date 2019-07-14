import random
import src.Config.NeatProperties as Props
import math


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

    def add(self, individual):
        self.members.append(individual)
        """removed becasue speciation redifines species borders  -so allowing more individuals in is acceptable"""
        # if len(self.members) > self.next_species_size:
        #     raise Exception("added too many individuals to species. max:", self.next_species_size)

    def step(self, mutation_record):
        if len(self.members) == 0:
            raise Exception("cannot step empty species")

        if self.next_species_size == 0:
            self.members = []
            return

        self._rank_species()
        elite_count = min(math.ceil(Props.ELITE_TO_KEEP * len(self.members)), self.next_species_size)
        self._cull_species()

        self._reproduce(mutation_record, elite_count)

        if len(self.members) != self.next_species_size:
            raise Exception("created next generation but population size(" + repr(
                len(self.members)) + ")is wrong should be:(" + repr(self.next_species_size) + ")")

        self._select_representative()
        self.age += 1

    def _reproduce(self, mutation_record, number_of_elite):
        # print("reproducing species(",self.get_species_type(),") of size",len(self.members),"with target member size=", self.next_species_size,end=", ")
        # print("number of elite:", number_of_elite, "num children to be created:",(self.next_species_size - number_of_elite))
        elite = self.members[:number_of_elite]
        children = []
        tries = 10 * (self.next_species_size - len(elite))

        while len(children) + len(elite) < self.next_species_size:
            parent1 = random.choice(self.members)
            parent2 = random.choice(self.members)

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
                raise Exception("Error: Species " + repr(self) + " failed to create enough healthy offspring")

        children.extend(elite)
        self.members = children

    def _rank_species(self):
        # note original checks for equal fitnesses and chooses one with more genes
        self.members.sort(key=lambda x: x.rank)

    def get_average_rank(self):
        return sum([indv.rank for indv in self.members]) / len(self.members)

    def _cull_species(self):
        if self.age < 3:
            return

        # print("culing species with", len(self.members), end="; ")
        surivors = math.ceil(Props.PERCENT_TO_REPRODUCE * len(self.members))
        self.members = self.members[:surivors]
        # print("after culling:", len(self.members))

    def _select_representative(self):
        self.representative = random.choice(self.members)

    def empty_species(self):
        self.members = []

    def set_next_species_size(self, species_size):
        self.next_species_size = species_size

    def sample_individual(self):
        index = random.randint(0, len(self.members) - 1)
        return self.members[index], index

    def get_individual_by_index(self,index):
        return self.members[index]

    def get_species_type(self):
        if len(self.members) > 0:
            return type(self.members[0])
