import copy
import time

from src.Analysis.EvolutionaryDataPlotter import plot_acc_vs_second, plot_histogram
from src.Config import Config
from src.NEAT.PopulationRanking import general_pareto_sorting
from src.Validation import Validation


class ParetoPopulation:

    def __init__(self):
        self.pareto_front = []
        self.candidates = []
        self.best_members = []
        self.worst_das = []

    def queue_candidate(self, candidate):
        self.candidates.append(copy.deepcopy(candidate))

    def update_pareto_front(self):
        """creates a new pareto front of best solutions by combining the previous
        pareto front with all of the new  individuals scored in the last generation"""

        self.best_members.append(self.get_highest_accuracy(1, check_set=self.candidates))
        if Config.evolve_data_augmentations:
            self.worst_das.append(self.get_worst_da_from_candidates())

        self.pareto_front = general_pareto_sorting(self.candidates + self.pareto_front, return_pareto_front_only=True)

        if len(self.pareto_front) == 0:
            raise Exception("pareto front empty after step")

        self.candidates = []

    def get_trained_best_network(self, num_augs=1):
        """selects the best network, and trains it fully"""
        best_graphs = self.get_highest_accuracy(num=max(len(self.best_members) - 8, 1), check_set=self.best_members)
        best = best_graphs[0]
        print("Fully training", Config.run_name, "reported acc:", best.fitness_values[0])

        augs = [x.data_augmentation_schemes[0] for x in best_graphs if len(x.data_augmentation_schemes) > 0]
        aug_names = set()
        unique_augs = []
        for aug in augs:
            name = repr(aug).split("Nodes:")[1].replace("'No_Operation'", "").replace("[]", "").replace('\\n',
                                                                                                        "").replace(",",
                                                                                                                    "").replace(
                '"', "").replace(" ", "")

            if name not in aug_names:
                aug_names.add(name)
                unique_augs.append(aug)
        unique_augs = unique_augs[:num_augs]
        return Validation.get_fully_trained_network(best, unique_augs, num_epochs=Config.num_epochs_in_full_train)

    def plot_fitnesses(self):
        """analysis tool which plots the scores of individuals in the pareto pop"""
        accuracies = [x.fitness_values[0] for x in self.pareto_front]
        num_objectives = len(self.pareto_front[0].fitness_values)
        if num_objectives == 1:
            plot_histogram(accuracies)
        elif num_objectives == 2:
            plot_acc_vs_second(accuracies, [x.fitness_values[1] for x in self.pareto_front])
        elif num_objectives == 3:
            pass
        else:
            raise Exception(">3 objectives")

    def get_highest_accuracy(self, num, plot_best=False, check_set=None):
        """gets the n highest performing individuals from a given set"""

        highest_acc = 0
        best_graph = None
        if check_set is None:
            check_set = self.pareto_front

        if num > 1:
            acc_sorted = sorted(check_set, key=lambda x: x.fitness_values[0], reverse=True)
            num_best_graphs = acc_sorted[:num]
            return num_best_graphs

        elif num == 1:

            for graph in check_set:
                if graph.fitness_values[0] > highest_acc:
                    highest_acc = graph.fitness_values[0]
                    best_graph = graph

            if plot_best:
                best_graph.plot_tree_with_graphvis("best acc graph in pareto population - acc=" + repr(highest_acc))

            return best_graph
        else:
            raise Exception("Number of graphs chosen is negative")

    def get_worst_da_from_candidates(self):
        """analysis tool"""
        worst = None
        worst_acc = 9999999
        for mod_graph in self.candidates:
            da = mod_graph.data_augmentation_schemes[0]
            if da.fitness_values[0] < worst_acc:
                worst = da
                worst_acc = da.fitness_values[0]
        return worst
