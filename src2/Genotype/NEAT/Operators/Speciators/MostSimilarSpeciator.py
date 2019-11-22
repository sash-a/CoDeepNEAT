import math
from typing import List, TYPE_CHECKING

from src2.Genotype.NEAT.Operators.Speciators.Speciator import Speciator
from src2.Configuration import config
from src2.Genotype.NEAT.Species import Species

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome


class MostSimilarSpeciator(Speciator):
    def speciate(self, specieses: List[Species]) -> None:
        self.adjust_speciation_threshold(len(specieses))
        individuals: List[Genome] = [member for spc in specieses for member in spc.members.values()]

        # print("rep:",specieses[0].representative)

        for spc in specieses:
            spc.clear()

        # if species[0].representative is None:
        # print("num species", len(specieses))
        # print("num members in spc[0]", len(specieses[0]))
        # print(specieses)

        for individual in individuals:
            best_fit_species = None
            best_distance = individual.distance_to(specieses[0].representative) + 1

            # find best species
            for species in specieses:
                distance = individual.distance_to(species.representative)
                if distance < best_distance:
                    best_distance = distance
                    best_fit_species = species

            if best_distance <= self.threshold:
                best_fit_species.add(individual)
            else:
                # none of the existing species were close enough for this individual
                # create a new species only if it is not more than the max number of species
                if len(specieses) < self.target_num_species:
                    specieses.append(Species(individual, self.mutator))
                else:
                    # Add individual to closest species
                    best_fit_species.add(individual)
                    self.threshold *= 1.1

    def adjust_speciation_threshold(self, n_species: int):
        if self.target_num_species == 1:
            return

        if n_species < self.target_num_species:
            new_dir = -1  # decrease thresh
        elif n_species > self.target_num_species:
            new_dir = 1  # increase thresh
        else:
            self.current_threshold_dir = 0
            return

        # threshold must be adjusted
        if new_dir != self.current_threshold_dir:
            # still not right - must have jumped over the ideal value adjust by base modification
            self.threshold = \
                min(
                    max(
                        config.species_distance_thresh_mod_min,
                        self.threshold + (new_dir * config.species_distance_thresh_mod_base)
                    ),
                    config.species_distance_thresh_mod_max
                )
        else:
            # still approaching the ideal value - exponentially speed up
            self.threshold *= math.pow(2, new_dir)

        self.current_threshold_dir = new_dir
