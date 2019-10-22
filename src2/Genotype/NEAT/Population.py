from typing import List, Dict, TYPE_CHECKING, Type

from NEAT import Genome, Species


class Population:
    def __init__(self):
        self.species: Species

    def speciate(self):
        pass

    def update_species_sizes(self):
        pass
