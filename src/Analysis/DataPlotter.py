import matplotlib.pyplot as plt
from src.Analysis import RuntimeAnalysis
import time

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
        plot_acc_vs_second(acc,second)
    else:
        pass

def plot_acc_vs_second(acc, second):
    global plot
    print("plotting",acc,"against",second)
    # if plot is None:
    #     #plt.ion()
    #     plt.pause(0.0001)
    #     plot = plt.scatter(acc, second)
    #     print("created plot obj",plot)
    #     plt.show()
    # else:
    #     plot.set_data(acc,second)
    #     plot.draw()
    plt.scatter(acc, second)
    plt.show()


def plot_histogram(acc):
    pass

def plot_generations():
    for generation in RuntimeAnalysis.generations:
        plot_objectives_at_gen(generation.generation_number)
        #time.sleep(1.5)



if __name__ == "__main__":
    RuntimeAnalysis.load_date_from_log_file("test",summary=False)
    plot_generations()