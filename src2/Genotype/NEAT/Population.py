from typing import List, Dict, TYPE_CHECKING, Iterable

from Genotype.NEAT.Genome import Genome
from Genotype.NEAT.Operators.Mutations.MutationRecord import MutationRecords
from Genotype.NEAT.Operators.PopulationRankers.PopulationRanker import PopulationRanker
from Genotype.NEAT.Operators.Speciators.Speciator import Speciator
from Genotype.NEAT.Species import Species


class Population:
    """Holds species, which hold individuals. Runs a single generation of the CoDeepNEAT evolutionary process."""
    ranker: PopulationRanker

    def __init__(self, individuals: List[Genome], mutation_record: MutationRecords, pop_size: int,
                 speciator: Speciator):
        # initial speciation process
        self.species: List[Species] = [Species(individuals[0])]
        for individual in individuals[1:]:
            self.species[0].add(individual)
        speciator.speciate(self.species)

        self.pop_size: int = pop_size
        self.mutation_record: MutationRecords = mutation_record
        self.speciator: Speciator = speciator

    def __iter__(self) -> Iterable[Genome]:
        return iter([member for spc in self.species for member in spc])

    def update_species_sizes(self):
        # TODO check the old code
        pass

    def step(self):
        # TODO delete a species if size is 0
        Population.ranker.rank(iter(self))
        self.update_species_sizes()

        for spc in self.species:
            spc.step(self.mutation_record)

        self.speciator.speciate(self.species)
