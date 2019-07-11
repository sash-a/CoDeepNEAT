from src.NEAT2.Species import Species
import src.Config.NeatProperties as Props


class Population:
    def __init__(self, individuals, population_size):
        self.species = []
        self.speciate(individuals)
        self.population_size = population_size
        self.speciation_threshold = 1  # TODO
        self.target_num_species = 4  # TODO

        self.current_threshold_dir = 1

    individuals = property(lambda self: self._get_all_individuals())

    def __iter__(self):
        return iter(self._get_all_individuals())

    def _get_all_individuals(self):
        individuals = []
        for species in self.species:
            individuals.extend(species.members)
        return individuals

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def speciate(self, individuals):
        for species in self.species:
            species.empty_species()

        """note origonal neat placed individuals in the first species they fit"""

        for individual in individuals:
            best_fit_species = None
            best_distance = individual.distance_to(self.species[0].representative) + 1

            """find best species"""
            for species in self.species:
                distance = individual.distance_to(species.representative)
                if distance < best_distance:
                    best_distance = distance
                    best_fit_species = species

            """fit individual somewhere"""
            if best_distance <= self.speciation_threshold:
                best_fit_species.add(individual)
            else:
                self.species.append(Species(individual))

        self.species = [spc for spc in self.species if spc.members]
        self.adjust_speciation_threshold()

    def adjust_speciation_threshold(self):
        if len(self.species) < self.target_num_species:
            new_dir = -1  # decrease thresh
        elif len(self.species) > self.target_num_species:
            new_dir = 1  # increase thresh
        else:
            new_dir = 0

        if new_dir != self.current_threshold_dir:
            self.num_species_mod = Props.SPECIES_DISTANCE_THRESH_MOD
        else:
            self.num_species_mod *= 2

        self.current_threshold_dir = new_dir
        self.speciation_thresh = max(0.001, self.speciation_thresh + (new_dir * self.num_species_mod))

    def update_species_sizes(self):
        """should be called before species.step()"""
        population_average_rank = self.get_average_rank()

        total_species_fitness = 0
        for species in self.species:
            species_average_rank = species.get_average_rank()
            species.fitness = species_average_rank / population_average_rank
            total_species_fitness += species.fitness

        for species in self.species:
            species_size = round(self.population_size * (species.fitness / total_species_fitness))
            species.set_next_species_size(species_size)

    def rank_population(self):
        individuals = self._get_all_individuals()
        individuals.sort(key=lambda x: x.fitness_values[0], reverse=True)
        for i, individual in enumerate(individuals):
            individual.rank = i

    def get_average_rank(self):
        individuals = self._get_all_individuals()
        return sum([indv.rank for indv in individuals]) / len(individuals)

    def step(self):
        self.rank_population()
        self.update_species_sizes()
        for species in self.species:
            species.step()

        self.adjust_speciation_threshold()
        self.speciate(self._get_all_individuals())
