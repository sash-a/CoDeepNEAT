import matplotlib.pyplot as plt
from src.Analysis import RuntimeAnalysis
from data import DataManager
import numpy as np
import os

plot = None


def plot_objectives_at_gen(generation):
    if len(RuntimeAnalysis.generations) <= generation:
        return
    generation = RuntimeAnalysis.generations[generation]
    acc = generation.accuracies
    second = generation.second_objective_values
    third = generation.third_objective_values

    if second is None or len(second) == 0:
        plot_histogram(acc)
    elif third is None or len(third) == 0:
        plot_acc_vs_second(acc, second)
    else:
        pass


def plot_acc_vs_second(acc, second):
    global plot
    plt.scatter(acc, second)
    plt.show()


def plot_histogram(acc):
    plt.hist(acc, bins=20)
    plt.show()


def plot_generations():
    for generation in RuntimeAnalysis.generations:
        plot_objectives_at_gen(generation.generation_number)

def get_gens_and_fitnesses(aggregation_type='max', fitness_index=0):
    gens = list(range(0, len(RuntimeAnalysis.generations)))
    if aggregation_type == 'max':
        fitness = [gen.get_max_of_objective(fitness_index) for gen in RuntimeAnalysis.generations]
    elif aggregation_type == 'avg':
        fitness = [gen.get_average_of_objective(fitness_index) for gen in RuntimeAnalysis.generations]
    else:
        raise ValueError('Only aggregation types allowed are avg and max, received' + str(aggregation_type))

    return gens, fitness


def plot_all_generations(aggregation_type='max', fitness_index=0, run_name='unnamed run'):
    gens, fitness = get_gens_and_fitnesses(aggregation_type,fitness_index)

    plt.ylim(0, 100)
    plt.scatter(gens, fitness)
    plt.plot(np.unique(gens), np.poly1d(np.polyfit(gens, fitness, 1))(np.unique(gens)))
    plt.title(aggregation_type + ' value of objectives ' + str(fitness_index) + ' per generation for ' + run_name)
    plt.show()

def plot_all_runs(aggregation_type='max', fitness_index=0, max_gens = 100, show_data = False):
    runs = set()
    for subdir, dirs, files in os.walk(os.path.join(DataManager.get_data_folder(),"runs")):
        sub = subdir.split("runs")[1][1:].split("\\")[0].split("/")[0]
        #print(sub)
        if sub == "":
            continue
        runs.add(sub)

    for run in runs:
        try:
            RuntimeAnalysis.load_date_from_log_file(run, summary=False)
            gens, fitness = get_gens_and_fitnesses(aggregation_type, fitness_index)
            if len(gens) > max_gens:
                gens = gens[:max_gens]
                fitness = fitness[:max_gens]

            plt.plot(np.unique(gens), np.poly1d(np.polyfit(gens, fitness, 1))(np.unique(gens)), label = run)
            if show_data:
                plt.scatter(gens,fitness, label = run)
        except:
            continue

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels)
    plt.xlabel("Generation")
    plt.ylabel("fitness"+ repr(fitness_index))
    plt.show()

    print(runs)

if __name__ == "__main__":
    run_name = 'module_retention_test'
    #RuntimeAnalysis.load_date_from_log_file(run_name, summary=False)
    #plot_all_generations('max', 0, run_name)
    plot_all_runs(aggregation_type= "max")
