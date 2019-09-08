import os
from data import DataManager
import src.Config.Config as Config
from src.Analysis.EvolutionaryDataPlotter import get_all_run_names
from src.NeuralNetwork.ModuleNet import create_nn
# generation = DataManager.load_generation_state(Config.run_name)
from src.Validation import DataLoader

"""a file to trigger the plotting of the top n graphs in each run inside data/runs"""

def plot_all_graphs(top_num=1):
    runs = get_all_run_names()

    for run in runs:
        plot_best_graphs(run, top_num=top_num)


def plot_best_graphs(run_name, top_num=1, view = True):
    # try:
    generation = DataManager.load_generation_state(run_name)
    # except Exception as e:
    #     print(e)
    #     return

    best_graphs = generation.pareto_population.get_highest_accuracy(num=top_num,
                                                                    check_set=generation.pareto_population.best_members)
    if top_num == 1:
        best_graphs = [best_graphs]
    for graph in best_graphs:
        sample, _ = DataLoader.sample_data(Config.get_device(), dataset=graph.dataset)
        model = create_nn(graph, sample, feature_multiplier=1)
        file_name = run_name + "_best" + repr(best_graphs.index(graph))
        graph.plot_tree_with_graphvis(view=view, file=os.path.join(DataManager.get_data_folder(),"runs",run_name, "Graphs",file_name) )


if __name__ == "__main__":
    plot_all_graphs(top_num=1)
