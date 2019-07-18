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
import operator

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
        # current_generation.evaluate(i)
        current_generation.evaluate(i)
        current_generation.step()
        print('completed gen', i, "in", (time.time() - gen_start_time), "elapsed time:", (time.time() - start_time),
              "\n\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Runs the CoDeepNEAT algorithm')

    parser.add_argument('--ignore', action='store_true', help='Uses all default args in src/Config/Config.py')

    parser.add_argument('-p', '--data-path', type=str, nargs='?', default=Config.data_path,
                        help='Directory to store the training and test data')
    parser.add_argument('--dataset', type=str, nargs='?', default=Config.dataset,
                        choices=['mnist', 'fassion_mnist', 'cifar'], help='Dataset to train with')
    parser.add_argument('-d', '--device', type=str, nargs='?', default=Config.device.type, choices=['cpu', 'cuda:0'],
                        help='Device to train on')
    parser.add_argument('--n-workers', type=int, nargs='?', default=Config.num_workers,
                        help='Number of workers to load each batch')
    parser.add_argument('-n', '--ngen', type=int, nargs='?', default=Config.num_generations,
                        help='Max number of generations to run CoDeepNEAT')
    parser.add_argument('-s', '--second', type=str,
                        nargs='*', default=(Config.second_objective, Config.second_objective_comparator),
                        help='Second objective name and lt or gt to indicate if a lower or higher value is better')
    parser.add_argument('-t', '--third', type=str, nargs='*',
                        default=(Config.third_objective, 'lt'),
                        help='Third objective name and lt or gt to indicate if a lower or higher value is better')
    parser.add_argument('-f', '--fake', action='store_true', help='Runs a dummy version, for testing')
    parser.add_argument('--protect', action='store_false', help='Protects from possible graph parsing errors')  # TODO?
    parser.add_argument('-g', '--graph-save', action='store_true', help='Saves the best graphs in a generation')

    args = parser.parse_args()

    if not args.ignore:
        print(args)

        if args.second is not None and len(args.second) not in (0, 2):
            parser.error('Either give no values for second, or two, not {}.'.format(len(args.second)))

        if args.third is not None and len(args.third) not in (0, 2):
            parser.error('Either give no values for third, or two, not {}.'.format(len(args.third)))

        Config.data_path = args.data_path
        Config.dataset = args.dataset
        Config.device = torch.device(args.device)
        Config.num_workers = args.n_workers
        Config.num_generations = args.ngen
        Config.second_objective, second_obj_comp = args.second
        Config.third_objective, third_obj_comp = args.third
        Config.dummy_run = args.fake
        Config.protect_parsing_from_errors = args.protect
        Config.save_best_graphs = args.graph_save

        if second_obj_comp == 'lt':
            Config.second_objective_comparator = operator.lt
        elif second_obj_comp == 'gt':
            Config.second_objective_comparator = operator.gt
        else:
            parser.error('Must have only lt or gt as the second arg of --second')

        if third_obj_comp == 'lt':
            Config.third_objective_comparator = operator.lt
        elif second_obj_comp == 'gt':
            Config.third_objective_comparator = operator.gt
        else:
            parser.error('Must have only lt or gt as the second arg of --third')

        print(Config.second_objective_comparator)


if __name__ == '__main__':
    main()
