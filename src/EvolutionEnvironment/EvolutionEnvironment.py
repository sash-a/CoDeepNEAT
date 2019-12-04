import os
import sys

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_2 = os.path.split(dir_path)[0]
sys.path.append(dir_path_1)
sys.path.append(dir_path_2)

from src.EvolutionEnvironment.Generation import Generation
import src.Config.Config as Config
from data import DataManager

import time
import argparse
import operator
import torch.multiprocessing as mp
import cv2

"""
Evolution Environment is static as there should only ever be one
Acts as the driver of current generation
"""


def main():
    parse_args()
    cv2.setNumThreads(0)
    mp.set_start_method('spawn', force=True)
    if Config.continue_from_last_run:
        continue_evolution_from_save_state(Config.run_name)
    else:
        run_evolution_from_scratch()


def run_evolution_from_scratch():
    evolve_from_generation(Generation())


def continue_evolution_from_save_state(run_name):
    generation = DataManager.load_generation_state(run_name)
    generation.update_rank_function()
    evolve_from_generation(generation)


def evolve_from_generation(generation):
    """continues an evolutionary run for the given generation"""

    if generation.generation_number == -1:
        print("evolving gen from scratch")
        start_gen = 0
    else:
        print("loading generation:", generation)
        # generations save after completing their step. before incrementing their generation number
        start_gen = generation.generation_number + 1

    start_time = time.time()

    if start_gen < Config.max_num_generations and not Config.fully_train:
        for i in range(start_gen, Config.max_num_generations):
            print('Running gen', i)
            gen_start_time = time.time()
            generation.evaluate(i)
            generation.step_evolution()
            print('completed gen', i, "in", (time.time() - gen_start_time), "elapsed time:", (time.time() - start_time),
                  "\n\n")
        print("finished evolving", Config.max_num_generations, "generations")

    if Config.fully_train:
        generation.pareto_population.get_trained_best_network(num_augs=Config.num_augs_in_full_train)


def parse_args():
    parser = argparse.ArgumentParser(description='Runs the CoDeepNEAT algorithm')

    parser.add_argument('--ignore', action='store_true', help='Uses all default args in src/Config/Config.py')

    parser.add_argument('-p', '--data-path', type=str, nargs='?', default=Config.data_path,
                        help='Directory to store the training and test data')
    parser.add_argument('--dataset', type=str, nargs='?', default=Config.dataset,
                        choices=['mnist', 'fashion_mnist', 'cifar10'], help='Dataset to train with')
    parser.add_argument('-d', '--device', type=str, nargs='?', default=Config.device, choices=['cpu', 'gpu'],
                        help='Device to train on')
    parser.add_argument('--n-gpus', type=int, nargs='?', default=Config.num_gpus,
                        help='The number of GPUs available, make sure that --device is not cpu or leave it blank')
    parser.add_argument('--n-workers', type=int, nargs='?', default=Config.num_workers,
                        help='Number of workers to load each batch')
    parser.add_argument('-n', '--ngen', type=int, nargs='?', default=Config.max_num_generations,
                        help='Max number of generations to run CoDeepNEAT')
    parser.add_argument('-s', '--second', type=str, nargs='*', default=(Config.second_objective, 'lt'),
                        help='Second objective name and lt or gt to indicate if a lower or higher value is better')
    parser.add_argument('-t', '--third', type=str, nargs='*', default=(Config.third_objective, 'lt'),
                        help='Third objective name and lt or gt to indicate if a lower or higher value is better')
    parser.add_argument('-f', '--fake', action='store_true', help='Runs a dummy version, for testing')

    args = parser.parse_args()

    if not args.ignore:
        print(args)

        if args.second is not None and len(args.second) not in (0, 2):
            parser.error('Either give no values for second, or two, not {}.'.format(len(args.second)))

        if args.third is not None and len(args.third) not in (0, 2):
            parser.error('Either give no values for third, or two, not {}.'.format(len(args.third)))

        Config.data_path = args.data_path
        Config.dataset = args.dataset
        Config.device = args.device
        Config.num_workers = args.n_workers
        Config.max_num_generations = args.ngen
        Config.num_gpus = args.n_gpus
        if len(args.second) == 2:
            Config.second_objective, second_obj_comp = args.second
        if len(args.second) == 2:
            Config.third_objective, third_obj_comp = args.third
        Config.dummy_run = args.fake

        if len(args.second) == 2:
            if second_obj_comp == 'lt':
                Config.second_objective_comparator = operator.lt
            elif second_obj_comp == 'gt':
                Config.second_objective_comparator = operator.gt
            else:
                parser.error('Must have only lt or gt as the second arg of --second')

        if len(args.second) == 2:
            if third_obj_comp == 'lt':
                Config.third_objective_comparator = operator.lt
            elif second_obj_comp == 'gt':
                Config.third_objective_comparator = operator.gt
            else:
                parser.error('Must have only lt or gt as the second arg of --third')


if __name__ == '__main__':
    main()
