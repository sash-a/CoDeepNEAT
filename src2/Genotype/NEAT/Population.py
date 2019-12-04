from __future__ import annotations

import heapq
from typing import List, Iterable, Optional, Union, TYPE_CHECKING

from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.MutationRecord import MutationRecords
from src2.Genotype.NEAT.Operators.PopulationRankers.PopulationRanker import PopulationRanker
from src2.Genotype.NEAT.Operators.Speciators.Speciator import Speciator
from src2.Genotype.NEAT.Species import Species
from src2.Visualisation import SpeciationVisualiser

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.DAGenome import DAGenome
    from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome


class Population:
    """Holds species, which hold individuals. Runs a single generation of the CoDeepNEAT evolutionary process."""
    ranker: PopulationRanker

    def __init__(self, individuals: List[Union[Genome, ModuleGenome, BlueprintGenome, DAGenome]],
                 mutation_record: MutationRecords, pop_size: int, speciator: Speciator):
        # initial speciation process
        self.speciator: Speciator = speciator

        # print("given ", len(individuals))

        self.species: List[Species] = [Species(individuals[0], speciator.mutator)]
        for individual in individuals[1:]:
            # print("adding mem",individual)
            self.species[0].add(individual)

        # print("mems in spc:",len(self.species[0]))

        if self.speciator.target_num_species >1:
            self.speciator.speciate(self.species)

        self.pop_size: int = pop_size
        self.mutation_record: MutationRecords = mutation_record

    def __iter__(self) -> Iterable[Genome]:
        return iter([member for spc in self.species for member in spc])

    def __len__(self):
        return len([member for spc in self.species for member in spc])

    def __getitem__(self, genome_id):
        for species in self.species:
            if genome_id not in species.members:
                continue

            return species[genome_id]

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
        if len(self.species) == 0:
            raise Exception("no living species")

        try:
            species_adj_ranks = [sum([mem.rank for mem in species]) / len(species) for species in self.species]
        except ZeroDivisionError as ze:
            raise Exception("dead species in population")

        pop_adj_rank = sum(species_adj_ranks)
        for spc_adj_rank, species in zip(species_adj_ranks, self.species):
            species_size = round(self.pop_size * spc_adj_rank / pop_adj_rank)
            species.next_species_size = species_size

    def step(self):
        Population.ranker.rank(iter(self))
        self.species: List[Species] = [species for species in self.species if species]  # Removing empty species
        self._update_species_sizes()
        # Removing empty species
        # self.species: List[Species] = [species for species in self.species if species.next_species_size == 0]

        for spc in self.species:
            spc.step(self.mutation_record)

        if self.speciator.target_num_species >1:
            self.speciator.speciate(self.species)
        self.species: List[Species] = [species for species in self.species if species]  # Removing empty species

    def end_step(self):
        """calls the end step of each member"""
        for member in self:
            member.end_step()

    def before_step(self):
        for member in self:
            member.before_step()

    def aggregate_fitness(self):
        for individual in self:
            individual.aggregate_fitness()

    def is_alive(self, genome_id) -> bool:
        for spc in self.species:
            if genome_id in spc.members.keys():
                return True

        return False

    def get_most_accurate(self, n=1, return_unit_as_list = False):
        if n == 1 and not return_unit_as_list:
            return heapq.nlargest(n, iter(self), key=lambda x: x.accuracy)[0]

        return heapq.nlargest(n,iter(self), key=lambda x: x.accuracy)

    def visualise(self, suffix=""):
        SpeciationVisualiser.visualise_specieses(self.species, suffix=suffix)
