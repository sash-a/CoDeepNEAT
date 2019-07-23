from src.NEAT.PopulationRanking import general_pareto_sorting
from src.Analysis.DataPlotter import plot_acc_vs_second,plot_histogram
import time

class ParetoPopulation():

    def __init__(self):
        self.pareto_front = []
        self.candidates = []

    def queue_candidate(self, candidate):
        self.candidates.append(candidate)

    def update_pareto_front(self):
        start_time = time.time()
        print("updating pareto pop from",len(self.candidates),"candidates and",len(self.pareto_front),"in front", end = " ")
        self.pareto_front = general_pareto_sorting(self.candidates + self.pareto_front, return_pareto_front_only=True)
        self.candidates = []
        print("after:",len(self.pareto_front),"in front time:", (time.time() - start_time))

    def plot_fitnesses(self):
        accuracies = [x.fitness_values[0] for x in self.pareto_front]
        num_objectives = len(self.pareto_front[0].fitness_values)
        if num_objectives == 1:
            plot_histogram(accuracies)
        elif num_objectives ==2:
            plot_acc_vs_second(accuracies, [x.fitness_values[1] for x in self.pareto_front])
        elif num_objectives == 3:
            pass
        else:
            raise Exception()

    def plot_all_in_pareto_front(self):
        for graph in self.pareto_front:
            graph.plot_tree_with_graphvis(file="fitnesses="+repr(graph.fitness_values))


    def get_highest_accuracy(self, print = False):
        highest_acc = 0
        best_graph = None

        for graph in self.pareto_front:
            if graph.fitness_values[0] > highest_acc:
                highest_acc = graph.fitness_values[0]
                best_graph = graph
        if print:
            best_graph.plot_tree_with_graphvis("best acc graph in pareto population - acc="+repr(highest_acc))
        return highest_acc
