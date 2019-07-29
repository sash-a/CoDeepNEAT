import sys
import os

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_2 = os.path.split(dir_path)[0]
sys.path.append(dir_path_1)
sys.path.append(dir_path_2)

import torch
from src.DataEfficiency import Net,DataEfficiency
from src.Validation.DataLoader import load_data
import math
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim


networks = [Net.StandardNet, Net.BatchNormNet, Net.DropOutNet]



def get_all_networks():
    all_networks = []
    for network_type in networks:
        for i in range(6):
            size = int(math.pow(2, i))
            model = network_type(size)
            all_networks.append(model)

    return all_networks

def plot_verbose_against_summarised():
    verbose_points = []
    summarised_points = []
    for model in get_all_networks():
        verbose,summarised = model.get_verbose_and_summarised_results()
        verbose_points.append(DataEfficiency.solve_for_learning_rate(verbose))
        summarised_points.append(DataEfficiency.solve_for_learning_rate(summarised))
        print("verbose:",verbose_points[-1], "summarised:", summarised_points[-1])


    plt.scatter(verbose_points, summarised_points, label = "verbose DE vs summarised DE")
    plt.xlabel("Verbose DE")
    plt.ylabel("Summarised DE")
    plt.show()


def plot_size_vs_DE(model_type, verbose):
    for i in range(6):
        size = int(math.pow(2, i))
        model = model_type(size)
        DE = DataEfficiency.solve_for_learning_rate(model.get_results(verbose))


if __name__ == "__main__":
    plot_verbose_against_summarised()