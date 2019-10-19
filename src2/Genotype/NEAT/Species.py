from src2.Genotype.NEAT.Operators import Selector
from src2.Genotype.NEAT.Operators.Mutations import Mutator


class Species:
    species_id = 0

    def __init__(self, representative, selector, mutator):
        self.id: int = Species.species_id
        Species.species_id += 1

        self.selector: Selector = selector
        self.mutator: Mutator = mutator

        self.representative = representative
        self.members = [representative]
        self.next_species_size = 1000
        self.fitness = -1
        self.tie_count = 0  # a count of how many ties there are for the top accuracy

    def fill(self):
        self.selector.before_selection()
        self._reproduce()

    def _reproduce(self):
        p1, p2 = self.selector.select(self.members)
