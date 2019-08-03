import time

from src.Analysis.DataPlotter import plot_acc_vs_second, plot_histogram
from src.NEAT.PopulationRanking import general_pareto_sorting
from src.Validation import Validation


class ParetoPopulation:
    def __init__(self):
        self.pareto_front = []
        self.candidates = []

    def queue_candidate(self, candidate):
        # print("queuing candidate:",candidate)
        self.candidates.append(candidate)

    def update_pareto_front(self):
        start_time = time.time()
        # print("updating pareto pop from",len(self.candidates),"candidates and",len(self.pareto_front),"in front", end = " ")
        self.pareto_front = general_pareto_sorting(self.candidates + self.pareto_front, return_pareto_front_only=True)
        # print("after:",len(self.pareto_front),"in front time:", (time.time() - start_time))
        # print("candidates:",repr(self.candidates))

        if len(self.pareto_front) == 0:
            raise Exception("pareto front empty after step")

        self.candidates = []
        # self.plot_fitnesses()
        # self.plot_all_in_pareto_front()
        # self.get_highest_accuracy(print=True)

    def get_best_network(self, num_augs=5):
        best_graphs = self.get_highest_accuracy(num=num_augs)
        best = best_graphs[0]
        augs = [x.data_augmentation_schemes[0] for x in best_graphs if len(x.data_augmentation_schemes) > 0]
        return Validation.get_fully_trained_network(best, augs)

    def plot_fitnesses(self):
        # print("lengths:" , repr([len(x.fitness_values) for x in self.pareto_front]))
        # print("pop:",self.pareto_front)
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

    def plot_all_in_pareto_front(self):
        for graph in self.pareto_front:
            graph.module_graph_root_node.plot_tree_with_graphvis(file="fitnesses=" + repr(graph.fitness_values))

    def get_highest_accuracy(self, num, print=False):
        highest_acc = 0
        best_graph = None

        if num > 1:
            acc_sorted = sorted(self.pareto_front, key=lambda x: x.fitness_values[0])
            num_best_graphs = acc_sorted[-num:]
            return num_best_graphs

        elif num == 1:

            for graph in self.pareto_front:
                if graph.fitness_values[0] > highest_acc:
                    highest_acc = graph.fitness_values[0]
                    best_graph = graph

            if print:
                best_graph.plot_tree_with_graphvis("best acc graph in pareto population - acc=" + repr(highest_acc))

            return best_graph
        else:
            raise Exception("Number of graphs chosen is negative")
