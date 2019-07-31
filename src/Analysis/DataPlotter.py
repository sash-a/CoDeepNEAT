import matplotlib.pyplot as plt
from src.Analysis import RuntimeAnalysis

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


def plot_all_generations(aggregation_type='max', fitness_index=0, run_name='unnamed run'):
    gens = list(range(0, len(RuntimeAnalysis.generations)))
    if aggregation_type == 'max':
        fitness = [gen.get_max_of_objective(fitness_index) for gen in RuntimeAnalysis.generations]
    elif aggregation_type == 'avg':
        fitness = [gen.get_average_of_objective(fitness_index) for gen in RuntimeAnalysis.generations]
    else:
        raise ValueError('Only aggregation types allowed are avg and max, received' + str(aggregation_type))

    plt.ylim(0, 100)
    plt.scatter(gens, fitness)
    plt.title(aggregation_type + ' value of objectives ' + str(fitness_index) + ' per generation for ' + run_name)
    plt.show()


if __name__ == "__main__":
    run_name = 'module_retention_test'
    RuntimeAnalysis.load_date_from_log_file(run_name, summary=False)
    plot_all_generations('avg', 0, run_name)
