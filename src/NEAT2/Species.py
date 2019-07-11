import random
import src.Config.NeatProperties as Props


class Species:
    def __init__(self, representative):
        # Possible extra attribs:
        # age, hasBest, noImprovement
        self.representative = representative
        self.members = [representative]
        self.next_species_size = -1
        self.fitness = -1

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def add(self, individual):
        self.members.append(individual)
        if len(self.members) > self.next_species_size:
            raise Exception("added too many individuals to species. max:", self.next_species_size)

    def step(self):
        self._rank_species()
        self._cull_species()
        self._reproduce()
        if len(self.members) != self.next_species_size:
            raise Exception("created next generation but population size(" + repr(
                len(self.members)) + ")is wrong should be:(" + repr(self.next_species_size) + ")")

        self._select_representative()

    def _reproduce(self):
        elite = int(Props.ELITE_TO_KEEP * len(self.members) / Props.PERCENT_TO_REPRODUCE)
        elite = self.members[:elite]
        children = []
        tries = 10 * (self.next_species_size - len(elite))
        while len(children) + len(elite) < self.next_species_size:
            parent1 = random.choice(self.members)
            parent2 = random.choice(self.members)
            if parent1 == parent2:
                continue

            best = parent1 if parent1.rank < parent2.rank else parent2
            worst = parent1 if parent1.rank > parent2.rank else parent2

            child = best.crossover(worst)
            if child is None:
                raise Exception("Error: cross over produced null child")

            if child.validate():
                children.append(child.mutate())

            tries -= 1
            if tries == 0:
                raise Exception("Error: Species " + repr(self) + " failed to create enough healthy offspring")

        self.members = elite.extend(children)

    def _rank_species(self):
        # note origonal checks for equal fitnesses and choses one with fewer genes
        self.members.sort(key=lambda x: x.rank)

    def get_average_rank(self):
        return sum([indv.rank for indv in self.members]) / len(self.members)

    def _cull_species(self):
        surivors = Props.PERCENT_TO_REPRODUCE * len(self.members)
        self.members = self.members[:surivors]

    def _select_representative(self):
        self.representative = random.choice(self.members)

    def empty_species(self):
        self.members = []

    def set_next_species_size(self, species_size):
        self.next_species_size = species_size
