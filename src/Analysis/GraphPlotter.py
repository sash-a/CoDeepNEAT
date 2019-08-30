import src.Config.Config as Config
from data import DataManager
from src.NeuralNetwork.ModuleNet import create_nn
from src.Analysis.DataPlotter import get_all_run_names

# generation = DataManager.load_generation_state(Config.run_name)
from src.Validation import DataLoader


def plot_all_graphs(top_num=1):
    runs = get_all_run_names()

    for run in runs:
        plot_best_graphs(run, top_num=top_num)


def plot_best_graphs(run_name, top_num = 1):
    try:
        generation = DataManager.load_generation_state(run_name)
    except:
        return

    best_graphs = generation.pareto_population.get_highest_accuracy(num=top_num,check_set=generation.pareto_population.best_members)

    for graph in best_graphs:
        sample, _ = DataLoader.sample_data(Config.get_device(), dataset=graph.dataset)
        model = create_nn(graph, sample, feature_multiplier=1)

        graph.plot_tree_with_graphvis(view=True, file=run_name + "_best" + repr(best_graphs.index(graph)))


if __name__ == "__main__":
    plot_all_graphs(top_num = 2)
