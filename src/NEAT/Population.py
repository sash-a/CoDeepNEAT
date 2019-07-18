from src.NEAT.Species import Species
import src.Config.NeatProperties as Props
from src.Config import Config
import math


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

        self.speciation_threshold = 1
        self.current_threshold_dir = 1

        self.mutation_record = MutationRecords(initial_mutations, max_node_id, max_innovation)

        self.rank_population_fn = rank_population_fn

        self.species = [Species(individuals[0])]
        self.species[0].members = individuals

    individuals = property(lambda self: self._get_all_individuals())

    def __iter__(self):
        return iter(self._get_all_individuals())

    def __repr__(self):
        return "population of type:" + repr(type(self.species[0].members[0]))

    def _get_all_individuals(self):
        individuals = []
        for species in self.species:
            individuals.extend(species.members)
        return individuals

    def get_num_species(self):
        return len(self.species)

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def speciate(self, individuals):
        for species in self.species:
            species.empty_species()

        """note original neat placed individuals in the first species they fit this places in the closest species"""

        for individual in individuals:
            best_fit_species = None
            best_distance = individual.distance_to(self.species[0].representative) + 1

            """find best species"""
            for species in self.species:
                distance = individual.distance_to(species.representative)
                if distance < best_distance:
                    best_distance = distance
                    best_fit_species = species

            if best_distance <= self.speciation_threshold:
                best_fit_species.add(individual)
            else:
                self.species.append(Species(individual))

        self.species = [spc for spc in self.species if spc.members]

    def adjust_speciation_threshold(self):
        if len(self.species) < self.target_num_species:
            new_dir = -1  # decrease thresh
        elif len(self.species) > self.target_num_species:
            new_dir = 1  # increase thresh
        else:
            self.current_threshold_dir = 0
            return

        """threshold must be adjusted"""

        if new_dir != self.current_threshold_dir:
            """still not right - must have jumped over the ideal value
                adjust by base modification
            """
            self.speciation_threshold = min(max(Props.SPECIES_DISTANCE_THRESH_MOD_MIN, self.speciation_threshold + (
                    new_dir * Props.SPECIES_DISTANCE_THRESH_MOD_BASE)),
                                            Props.SPECIES_DISTANCE_THRESH_MOD_MAX)
        else:
            """still approaching the ideal value - exponentially speed up"""
            self.speciation_threshold *= math.pow(2, new_dir)

        # print("\tsetting new spec thresh to:",self.speciation_threshold,type(self._get_all_individuals()[0]), "num species:",len(self.species),"target:", self.target_num_species , "thresh:",self.speciation_threshold, "new dir:",new_dir, "old dir:",self.current_threshold_dir)
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

    def step(self):
        # print("stepping population of",type(self._get_all_individuals()[0]))
        self.rank_population_fn(self._get_all_individuals())
        # print(self,"ranked individuals")
        self.update_species_sizes()

        for species in self.species:
            species.step(self.mutation_record)

        self.adjust_speciation_threshold()
        individuals = self._get_all_individuals()
        self.speciate(individuals)


def single_objective_rank(individuals):
    individuals.sort(key=lambda indv: (0 if not indv.fitness_values else indv.fitness_values[0]), reverse=True)
    for i, individual in enumerate(individuals):
        individual.rank = i + 1


def cdn_pareto_front(individuals):
    individuals.sort(key=lambda indv: indv.fitness_values[0], reverse=True)

    pf = [individuals[0]]  # pareto front populated with best individual in primary objective

    for indv in individuals[1:]:
        if Config.second_objective_comparator(indv.fitness_values[1], pf[-1].fitness_values[1]):
            pf.append(indv)

    return pf


def cdn_rank(individuals):
    # print("called rank for",type(individuals[0]))
    for indv in individuals:
        if not indv.fitness_values:
            indv.fitness_values = [0, 0]  # TODO make sure second obj must be maximized

    ranked_individuals = []

    fronts = []
    while individuals:
        pf = cdn_pareto_front(individuals)
        fronts.append(pf)
        individuals = individuals[len(pf):]  # TODO does this drop an individual?
        ranked_individuals.extend(pf)

    for i, indv in enumerate(ranked_individuals):
        # print("ranking", ranked_individuals[0], "to",i)
        indv.rank = i + 1
    return fronts

def nsga_rank(individuals):
    fronts = general_pareto_sorting(individuals)

    rank = 1

    for front in fronts:
        """rank is firstly based on which front the indv is in"""
        distances = {}
        for objective in range(len(individuals[0].fitness_values)):
            """estimate density by averaging the two nearest along each objective axis, then combining each distance"""
            objective_sorted  = sorted(front, key = lambda x: x.fitness_values[objective], reverse=True)

            for i, indv in enumerate(objective_sorted):
                distance = ((abs(objective_sorted[i] - objective_sorted[i+1]) if i<len(objective_sorted)-1 else 0) + (abs(objective_sorted[i] - objective_sorted[i-1]) if i>0 else 0))/ (2 if i>0 and i<len(sorted)-1 else 1)
                distance = math.pow(distance,2)
                if i == 0:
                    distances[indv]=[]
                distances[indv].append(distance)

        distance_sorted = sorted(front, key = lambda x: sum(distances[x]))
        for indv in distance_sorted:
            indv.rank = rank
            rank += 1

def general_pareto_sorting(individuals):
    """takes in a list of individuals and returns a list of fronts, each being a list of individuals"""
    fronts = [[]]
    dominations = {}
    domination_counts = {}
    for indv in individuals:
        dominated_count = 0
        domination_by_indv = []
        for comparitor in individuals:
            if indv == comparitor:
                continue
            if check_domination(indv,comparitor):
                #print(indv.fitness_values,"dominated",comparitor.fitness_values)
                domination_by_indv.append(comparitor)
            elif check_domination(comparitor,indv):
                #print(indv.fitness_values,"was dominated by",comparitor.fitness_values)
                dominated_count+=1
        if dominated_count == 0:
            #print("found indv which is not dominated by anyone")
            fronts[0].append(indv)
        #print()

        dominations[indv] = domination_by_indv
        domination_counts[indv] = dominated_count

    #print("found front 0:",fronts[0])

    front_number = 0
    while True:
        next_front = set()
        for leader in fronts[front_number]:
            for dominated_individual in dominations[leader]:
                domination_counts[dominated_individual]-=1
                if domination_counts[dominated_individual] == 0:
                    next_front.add(dominated_individual)

        if len(next_front) ==0:
            break
        fronts.append(next_front)
        front_number+=1

    return fronts

def check_domination(domination_candidate, comparitor):
    """checks if the domination candidate dominates the comparitor"""
    for i in range(len(domination_candidate.fitness_values)):
        if comparitor.fitness_values[i] > domination_candidate.fitness_values[i]:
            return False
    return True



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.NEAT.Genome import Genome
    import random
    fake_individuals = []
    for i in range(1000):
        fake_individuals.append(Genome([],[]))
        fake_individuals[-1].fitness_values = [random.random(),random.random()]
    fronts = cdn_rank(fake_individuals)
    #print(len(fronts))
    for front in fronts:
        sorted_front = sorted(front,key=lambda indv: indv.fitness_values[0])
        x = [indv.fitness_values[0] for indv in sorted_front]
        y = [indv.fitness_values[1] for indv in sorted_front]
        plt.plot(x,y)
    plt.show()



