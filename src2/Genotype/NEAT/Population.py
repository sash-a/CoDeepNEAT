from typing import List, Iterable, Optional

from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.MutationRecord import MutationRecords
from src2.Genotype.NEAT.Operators.PopulationRankers.PopulationRanker import PopulationRanker
from src2.Genotype.NEAT.Operators.Speciators.Speciator import Speciator
from src2.Genotype.NEAT.Species import Species


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

    def __len__(self):
        return len(list(self))

    def __getitem__(self, item):
        for species in self.species:
            if item not in species.members:
                continue

            return species[item]

        return None

    def get_species_by_id(self, species_id: int) -> Optional[Species]:
        for species in self.species:
            if species.id == species_id:
                return species

        return None

    def _update_species_sizes(self):
        """
        Setting the number of children that each species will produce. Based on its rank and how many members are
        already in the species
        """
        # TODO NEAT speciates based on fitness, how does rank (the way I am doing it) affect the sharing function?
        species_adj_ranks = [sum([mem.rank for mem in species]) / len(species) for species in self.species]

        pop_adj_rank = sum(species_adj_ranks)
        for spc_adj_rank, species in zip(species_adj_ranks, self.species):
            species_size = round(self.pop_size * spc_adj_rank / pop_adj_rank)
            species.next_species_size = species_size

    def step(self):
        Population.ranker.rank(iter(self))
        self._update_species_sizes()
        self.species = [species for species in self.species if species.next_species_size != 0]  # Removing empty species

        for spc in self.species:
            spc.step(self.mutation_record)

        self.speciator.speciate(self.species)

    def is_alive(self, genome_id) -> bool:

        for spc in self.species:
            if genome_id in spc.members.keys():
                return True

        return False
