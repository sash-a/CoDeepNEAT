import os

import math

from src.Config.Config import Config
from src.NEAT.Species import Species
import matplotlib.pyplot as plt
from data import DataManager



class MutationRecords:
    def __init__(self, initial_mutations, current_max_node_id, current_max_conn_id):
        self.mutations = initial_mutations
        self._next_node_id = current_max_node_id
        self._next_conn_id = current_max_conn_id

    def exists(self, mutation):
        return mutation in self.mutations

    def add_mutation(self, mutation):
        if type(mutation) == tuple:
            # Making sure tuple of ints
            for x in mutation:
                if not isinstance(x, int):
                    raise TypeError('Incorrect type passed to mutation: ' + mutation)

            self.mutations[mutation] = self.get_next_connection_id()
            return self._next_conn_id

        elif type(mutation) == int:
            self.mutations[mutation] = self.get_next_node_id()
            return self._next_node_id
        else:
            raise TypeError('Incorrect type passed to mutation: ' + mutation)

    def get_next_node_id(self):
        self._next_node_id += 1
        return self._next_node_id

    def get_next_connection_id(self):
        self._next_conn_id += 1
        return self._next_conn_id


class Population:
    def __init__(self, individuals, rank_population_fn, initial_mutations, population_size, max_node_id, max_innovation,
                 target_num_species):

        self.population_size = population_size
        self.target_num_species = target_num_species

        self.speciation_threshold = 2.3
        self.current_threshold_dir = 1

        if target_num_species == 1:
            self.speciation_threshold = float('inf')

        self.mutation_record = MutationRecords(initial_mutations, max_node_id, max_innovation)

        self.rank_population_fn = rank_population_fn

        self.species = [Species(individuals[0])]
        self.species[0].members = individuals

    individuals = property(lambda self: self._get_all_individuals())

    def __iter__(self):
        return iter(self._get_all_individuals())

    def __repr__(self):
        return 'population of type:' + repr(type(self.species[0].members[0]))

    def _get_all_individuals(self):
        individuals = []
        for species in self.species:
            individuals.extend(species.members)
        return individuals

    def __len__(self):
        return len(self._get_all_individuals())

    def __getitem__(self, item):
        return self._get_all_individuals()[item]

    def get_num_species(self):
        return len(self.species)

    def find_individual(self, indv):
        for i in range(len(self.species)):
            if indv in self.species[i].members:
                return i, self.species[i].members.index(indv)

        return -1, -1

    def speciate(self, individuals):
        for species in self.species:
            species.empty_species()

        # note original neat placed individuals in the first species they fit this places in the closest species
        for individual in individuals:
            if Config.speciation_overhaul:
                best_fit_species = None
                best_distance = individual.distance_to(self.species[0].representative) + 1

                # find best species
                for species in self.species:
                    distance = individual.distance_to(species.representative)
                    if distance < best_distance:
                        best_distance = distance
                        best_fit_species = species

                if best_distance <= self.speciation_threshold:
                    best_fit_species.add(individual)
                else:
                    """none of the existing species were close enough for this individual"""
                    if len(self.species) < self.target_num_species:
                        """create a new species and add this individual to it"""
                        self.species.append(Species(individual))
                    else:
                        """there are already the maximum number of species. add individual to closest species
                            the species threshold is too low"""
                        best_fit_species.add(individual)
                        self.speciation_threshold *= 1.1
            else:
                found = False
                for spc in self.species:
                    if individual.distance_to(spc.representative) <= self.speciation_threshold:
                        spc.add(individual)
                        found = True
                        break

                if not found:
                    self.species.append(Species(individual))

        self.species = [spc for spc in self.species if spc.members]

    def adjust_speciation_threshold(self):
        if self.target_num_species == 1:
            return

        if len(self.species) < self.target_num_species:
            new_dir = -1  # decrease thresh
        elif len(self.species) > self.target_num_species:
            new_dir = 1  # increase thresh
        else:
            self.current_threshold_dir = 0
            return

        # threshold must be adjusted
        if Config.speciation_overhaul:
            if new_dir != self.current_threshold_dir:
                # still not right - must have jumped over the ideal value adjust by base modification
                self.speciation_threshold = \
                    min(max(Config.species_distance_thresh_mod_min, self.speciation_threshold + (
                            new_dir * Config.species_distance_thresh_mod_base)), Config.species_distance_thresh_mod_max)
            else:
                # still approaching the ideal value - exponentially speed up
                self.speciation_threshold *= math.pow(2, new_dir)
        else:
            self.speciation_threshold += max(Config.species_distance_thresh_mod_min,
                                             Config.species_distance_thresh_mod_base * new_dir)

        self.current_threshold_dir = new_dir

    def update_species_sizes(self):
        """should be called before species.step()"""
        population_average_rank = self.get_average_rank()
        if population_average_rank == 0:
            raise Exception("population", self, "has an average rank of 0")

        total_species_fitness = 0
        for species in self.species:
            species_average_rank = species.get_average_rank()
            species.fitness = species_average_rank / population_average_rank
            total_species_fitness += species.fitness

        for species in self.species:
            species_size = round(self.population_size * (species.fitness / total_species_fitness))
            species.set_next_species_size(species_size)

    def get_average_rank(self):
        individuals = self._get_all_individuals()
        if len(individuals) == 0:
            raise Exception("no individuals in population", self, "cannot get average rank")
        return sum([indv.rank for indv in individuals]) / len(individuals)

    def step(self, generation = None):
        self.rank_population_fn(self._get_all_individuals())
        self.update_species_sizes()

        for species in self.species:
            species.step(self.mutation_record)

        self.adjust_speciation_threshold()
        individuals = self._get_all_individuals()
        self.speciate(individuals)
        # self.plot_species_spaces(generation)
        # self.plot_all_representatives()

    def plot_species_spaces(self, generation):
        if self.target_num_species == 1:
            return
        relative_individual = self.species[0].members[0]

        for spec in self.species:
            rep = spec.representative
            tops, atts = [], []
            for indv in spec.members:
                if indv == rep:
                    continue
                tops.append(relative_individual.get_topological_distance(indv))
                atts.append(relative_individual.get_attribute_distance(indv))
            plt.scatter(tops, atts, label="Species:" + repr(self.species.index(spec)))
            plt.scatter(relative_individual.get_topological_distance(rep),
                        relative_individual.get_attribute_distance(rep),
                        label="Species:" + repr(self.species.index(spec)) + "rep")
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles, labels)
        plt.xlabel("Topology")
        plt.ylabel("Attribute")
        plt.title("gen:" + repr(generation.generation_number))
        plt.show()

    def plot_all_representatives(self):
        graph = None
        for spec in self.species:
            graph = spec.representative.plot_tree_with_graphvis(graph= graph, return_graph_obj= True, view= False,
                                                                node_prefix= repr(self.species.index(spec)) + "_")
        file = os.path.join(DataManager.get_Graphs_folder(), "reps")
        graph.render(file, view=True)
