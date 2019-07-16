import sys
import os

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_2 = os.path.split(dir_path)[0]
sys.path.append(dir_path_1)
sys.path.append(dir_path_2)

from src.EvolutionEnvironment.Generation import Generation
from src.Analysis import RuntimeAnalysis
import src.Config.Config as Config

import torch
import time
import argparse

"""
Evolution Environment is static as there should only ever be one
Acts as the driver of current generation
"""


def main():
    parse_args()

    current_generation = Generation()
    RuntimeAnalysis.configure(log_file_name="test")
    start_time = time.time()

    for i in range(Config.num_generations):
        print('Running gen', i)
        gen_start_time = time.time()
        current_generation.evaluate(i)
        current_generation.step()
        print('completed gen', i, "in", (time.time() - gen_start_time), "elapsed time:", (time.time() - start_time),
              "\n\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Runs the CoDeepNEAT algorithm')

    parser.add_argument('-p', '--data_path', type=str, nargs='?', default=Config.data_path,
                        help='Directory to store the training and test data')
    parser.add_argument('-d', '--device', type=str, nargs='?', default=Config.device,
                        help='Device to train on e.g cuda:0')
    parser.add_argument('-n', '--ngen', type=int, nargs='?', default=Config.num_generations,
                        help='Max number of generations to run CoDeepNEAT')

    args = parser.parse_args()

    print(args)

    Config.data_path = args.data_path
    Config.device = torch.device(args.device)
    Config.num_generations = args.ngen


if __name__ == '__main__':
    main()
