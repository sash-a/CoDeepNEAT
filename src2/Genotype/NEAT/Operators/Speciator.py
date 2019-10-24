from abc import ABC, abstractmethod
from functools import reduce
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from Genotype.NEAT.Species import Species
    from Genotype.NEAT.Genome import Genome


class Speciator(ABC):
    @abstractmethod
    def speciate(self, species: List[Species], threshold: float, target_num_species: int) -> None:
        pass


class NEATSpeciator(Speciator):
    def speciate(self, species: List[Species], threshold: float, target_num_species: int) -> None:
        individuals: List[Genome] = [member for spc in species for member in spc.members]

        for individual in individuals:
            found = False
            for spc in species:
                if individual.distance_to(spc.representative) <= threshold:
                    spc.add(individual)
                    found = True
                    break

            if not found:
                species.append(Species(individual))


class MostSimilarSpeciator(Speciator):
    def speciate(self, species: List[Species], threshold: float, target_num_species: int) -> None:
        individuals: List[Genome] = [member for spc in species for member in spc.members]

        for individual in individuals:
            best_fit_species = None
            best_distance = individual.distance_to(species[0].representative) + 1

            # find best species
            for species in species:
                distance = individual.distance_to(species.representative)
                if distance < best_distance:
                    best_distance = distance
                    best_fit_species = species

            if best_distance <= threshold:
                best_fit_species.add(individual)
            else:
                # none of the existing species were close enough for this individual
                # create a new species only if it is not more than the max number of species
                if len(species) < target_num_species:
                    species.append(Species(individual))
                else:
                    # Add individual to closest species
                    best_fit_species.add(individual)
                    threshold *= 1.1  # TODO this change is not saved between generations
