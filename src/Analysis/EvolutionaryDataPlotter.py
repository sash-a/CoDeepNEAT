import heapq
import os

import matplotlib.pyplot as plt
import numpy as np
from data import DataManager

from src.Analysis import Logger

"""
file which reads generation by generation data containing info on the 
accuracy of every evaluated network, as well as their scores on subsequent objectives.

Data plotter aggregates each generations scores into one, either by using max, or average, or average of the top n
Data plotter then plots these aggregated scores at each generation for multiple runs
"""

plot = None


def plot_objectives_at_gen(generation):
    if len(Logger.generations) <= generation:
        return
    generation = Logger.generations[generation]
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
    for generation in Logger.generations:
        plot_objectives_at_gen(generation.generation_number)


def get_gens_and_fitnesses(aggregation_type='max', fitness_index=0, num_top=5):
    gens = list(range(0, len(Logger.generations)))
    if aggregation_type == 'max':
        fitness = [gen.get_max_of_objective(fitness_index) for gen in Logger.generations]
    elif aggregation_type == 'avg':
        fitness = [gen.get_average_of_objective(fitness_index) for gen in Logger.generations]
    elif aggregation_type == 'top':
        fitness = []
        for gen in Logger.generations:
            fitness.append(sum(heapq.nlargest(num_top, gen.objectives[fitness_index])) / num_top)
    else:
        raise ValueError('Only aggregation types allowed are avg and max, received' + str(aggregation_type))

    return gens, fitness


def plot_all_generations(aggregation_type='max', fitness_index=0, run_name='unnamed run'):
    gens, fitness = get_gens_and_fitnesses(aggregation_type, fitness_index)

    plt.ylim(0, 100)
    plt.scatter(gens, fitness)
    plt.plot(np.unique(gens), np.poly1d(np.polyfit(gens, fitness, 1))(np.unique(gens)))
    plt.title(aggregation_type + ' value of objectives ' + str(fitness_index) + ' per generation for ' + run_name)
    plt.show()


def get_all_run_names():
    runs = set()
    for subdir, dirs, files in os.walk(os.path.join(DataManager.get_data_folder(), "runs")):
        sub = subdir.split("runs")[1][1:].split("\\")[0].split("/")[0]
        if sub == "":
            continue
        runs.add(sub)

    return runs


def get_all_runs(aggregation_type='max', num_top=5, fitness_index=0, max_gens=1000):
    runs = get_all_run_names()

    runs_data = {}

    for run in runs:
        try:
            Logger.load_date_from_log_file(run, summary=False)
            gens, fitness = get_gens_and_fitnesses(aggregation_type, fitness_index, num_top=num_top)
            if len(gens) > max_gens:
                gens = gens[:max_gens]
                fitness = fitness[:max_gens]

            runs_data[run] = (gens, fitness)
        except:
            pass

    return runs_data


def get_run_groups(aggregation_type='max', num_top=5, fitness_index=0, max_gens=1000, include_deterministic_runs=True,
                   include_cross_species_runs=True):
    runs = get_all_runs(aggregation_type=aggregation_type, num_top=num_top, fitness_index=fitness_index,
                        max_gens=max_gens)
    groups = {}
    for run in runs.keys():
        group_run_name = get_run_group_name(run, include_deterministic_runs, include_cross_species_runs)
        if group_run_name not in groups:
            groups[group_run_name] = []
        groups[group_run_name].append(runs[run])

    return groups


def get_run_group_name(run_name, include_deterministic_runs=True, include_cross_species_runs=True):
    run_name = run_name.replace("da", "$")

    group_run_name = run_name.replace("_d", "") if include_deterministic_runs else run_name
    group_run_name = group_run_name.replace("_c", "") if include_cross_species_runs else group_run_name

    group_run_name = group_run_name.replace("$", "da")

    if group_run_name[-1].isdigit():
        group_run_name = group_run_name[:-1]

    if group_run_name in name_overrides:
        return name_overrides[group_run_name]

    return group_run_name


def get_run_boundries(aggregation_type='max', num_top=5, fitness_index=0, max_gens=1000,
                      include_deterministic_runs=True, smooth_boundries=True):
    run_groups = get_run_groups(aggregation_type=aggregation_type, num_top=num_top, fitness_index=fitness_index,
                                max_gens=max_gens, include_deterministic_runs=include_deterministic_runs)
    boundires = {}
    counts = {}

    for group_name in run_groups.keys():
        group = run_groups[group_name]
        fitnesses = [f for (g, f) in group]

        if len(fitnesses) < 2:
            """need at least 2 runs to get boundires"""
            continue
        """can get a boundry up till the second longest run, need 2 for a boundry"""
        max_num_gens = len(sorted(fitnesses, key=lambda x: len(x))[-2])
        mins = []
        maxes = []

        for i in range(max_num_gens):
            elements = [x[i] for x in fitnesses if len(x) > i]
            mins.append(min(elements))
            maxes.append(max(elements))

        if smooth_boundries:
            mins = get_rolling_averages(mins)
            maxes = get_rolling_averages(maxes)

        boundires[group_name] = (mins, maxes)
        counts[group_name] = len(group)
    return boundires, counts


def plot_all_runs(aggregation_type='max', num_top=5, fitness_index=0, max_gens=1000, show_data=False,
                  stay_at_max=True, line_graph=True, show_best_fit=False, show_smoothed_data=False,
                  show_boundires=True, smooth_boundries=True, show_data_in_boundries=True,
                  colour_group_run_lines_same=True):
    colours = {}
    if show_boundires:
        boundires, counts = get_run_boundries(aggregation_type=aggregation_type, num_top=num_top,
                                              fitness_index=fitness_index, max_gens=max_gens,
                                              smooth_boundries=smooth_boundries)
        for group_name in boundires.keys():
            mins, maxs = boundires[group_name]
            gens = [x for x in range(len(mins))]

            plot = plt.fill_between(gens, mins, maxs, alpha=0.4, label=group_name + ", n=" + repr(counts[group_name]))
            colours[group_name] = [max(min(x * 1.5, 1), 0) for x in plot.get_facecolor()[0]]

    runs = get_all_runs(aggregation_type=aggregation_type, num_top=num_top, fitness_index=fitness_index,
                        max_gens=max_gens)
    labels_used = set()
    for run in runs.keys():
        if show_boundires and not show_data_in_boundries:
            group_name = get_run_group_name(run)
            if group_name in boundires:
                continue
        gens, fitness = runs[run]

        aggregated = None
        if stay_at_max:
            aggregated = [max(fitness[:i + 1]) for i in range(len(fitness))]
        elif show_best_fit:
            gens = np.unique(gens)
            aggregated = np.poly1d(np.polyfit(gens, fitness, 1))(np.unique(gens))
        elif show_smoothed_data:
            aggregated = get_rolling_averages(fitness)
        group_name = get_run_group_name(run)
        colour = None
        if colour_group_run_lines_same:
            if show_boundires and show_data_in_boundries:
                if group_name in colours:
                    colour = colours[group_name]
        if show_data:
            if colour_group_run_lines_same:
                label = group_name if group_name not in labels_used else None
                labels_used.add(label)
            else:
                label = run

            if line_graph:
                p = plt.plot(gens, fitness, label=label, c=colour)
            else:
                p = plt.scatter(gens, fitness, label=label, c=colour)
            if aggregated is not None:
                plt.plot(gens, aggregated, c=p[0].get_color())

        else:
            if aggregated is not None:
                plt.plot(gens, aggregated, label=run, c=colour)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels)

    plt.set_cmap('gray')

    plt.xlabel("Generation")
    ylabel = "fitness " + repr(fitness_index) if fitness_index > 0 else "accuracy (%)"
    plt.ylabel(ylabel)
    title = aggregation_type + (" " + repr(num_top) if aggregation_type == "top" else "") + " fitness"
    plt.title("Top 5 accuracy of base and ModMax")

    # plt.show()
    plt.savefig('MMvsBase', dpi=300)


def get_rolling_averages(data, alpha=0.65):
    smoothed = []
    for point in data:
        if len(smoothed) == 0:
            smoothed.append(point)
        else:
            a = alpha if len(smoothed) > 10 else pow(alpha, 1.5)
            smooth = smoothed[-1] * a + point * (1 - a)
            smoothed.append(smooth)
    return smoothed


name_overrides = {"mm": "Modmax CDN", "mms": "Elite CDN", "mms_10E": "Elite CDN 10E", "base": "CDN",
                  "base_10E": "CDN 10E", "spc": "SPCDN", "base_da": "DACDN", "mms_da": "Elite DACDN",
                  "max": "max fitness aggregation CDN", "modret": "module retention CDN",
                  "mm_globmut": "ModMax with Global Mutation Adjustment",
                  "mms_globmut": "Elite CDN with Global Mutation Adjustment",
                  "mm_breed": "ModMax CDN with Node Breeding", "mms_breed": "Elite CDN with Node Breeding"}

if __name__ == "__main__":
    # style.use('fivethirtyeight')
    plot_all_runs(aggregation_type="top", num_top=5, show_data=True, show_best_fit=False, show_smoothed_data=False,
                  stay_at_max=False, show_boundires=True, smooth_boundries=False, show_data_in_boundries=True,
                  colour_group_run_lines_same=True, max_gens=30)
