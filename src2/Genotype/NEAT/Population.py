from typing import List, Dict, TYPE_CHECKING, Type

from Genotype.NEAT.Operators.Mutations.Mutator import Mutator
from Genotype.NEAT.Operators.RepresentativeSelector import RepresentativeSelector
from Genotype.NEAT.Operators.Selector import Selector
from Genotype.NEAT.Operators.Speciator import Speciator
from Genotype.NEAT.Species import Species


class Population:
    def __init__(self, selector: Selector, mutator: Mutator, representative_selector: RepresentativeSelector,
                 speciator: Speciator):
        self.species: List[Species] = []

        # This will be set three times, once by BP, once by modules, once by DAs.
        # TODO init this in main
        Species.selector = selector
        Species.mutator = mutator
        Species.representative = representative_selector

        self.speciator: Speciator = speciator

    def speciate(self):
        self.speciator.speciate(self.species)

    def update_species_sizes(self):
        pass
