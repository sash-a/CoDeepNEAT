from typing import List, Dict, TYPE_CHECKING, Type

from Genotype.NEAT.Operators.Mutations.Mutator import Mutator
from Genotype.NEAT.Operators.RepresentativeSelector import RepresentativeSelector
from Genotype.NEAT.Operators.Selector import Selector
from Genotype.NEAT.Operators.Speciator import Speciator
from Genotype.NEAT.Species import Species


class Population:
    speciator: Speciator

    def __init__(self, selector: Selector, mutator: Mutator, representative_selector: RepresentativeSelector,
                 speciator: Speciator):
        self.species: List[Species] = []

        # This will be set three times, once by BP, once by modules, once by DAs.
        # TODO find a better place to init this stuff
        Species.selector = selector
        Species.mutator = mutator
        Species.representative = representative_selector

        Population.speciator = speciator

    def speciate(self):
        Population.speciator.speciate(self.species, 1, 4)

    def update_species_sizes(self):
        pass
