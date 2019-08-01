import sys
import os

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

from src.Validation import Validation

"""
Evolution Environment is static as there should only ever be one
Acts as the driver of current generation
"""


def main():

    parse_args()
    mp.set_start_method('spawn', force=True)
    if Config.continue_from_last_run:
        try:
            continue_evolution_from_save_state(Config.run_name)
        except Exception as e:
            print(e)
            run_evolution_from_scratch()
    else:
        run_evolution_from_scratch()


def run_evolution_from_scratch():
    evolve_generation(Generation())

def continue_evolution_from_save_state(run_name):
    evolve_generation(DataManager.load_generation_state(run_name))

def evolve_generation(generation):
    #generation.pareto_population.plot_fitnesses()
    #generation.pareto_population.plot_all_in_pareto_front()
    #print("highest acc so far:",generation.pareto_population.get_highest_accuracy(print=True).fitness_values[0])
    if generation.generation_number == -1:
        print("evolving gen from scratch")
        start_gen = 0
    else:
        print("continueing evolution of generation:",generation)
        start_gen = generation.generation_number + 1 #genertions save after completing their step. before incremeting their generation number

    start_time = time.time()

    if start_gen < Config.max_num_generations:
        for i in range(start_gen, Config.max_num_generations):
            print('Running gen', i)
            gen_start_time = time.time()
            # current_generation.evaluate(i)
            generation.evaluate(i)
            generation.step()
            print('completed gen', i, "in", (time.time() - gen_start_time), "elapsed time:", (time.time() - start_time),
                  "\n\n")
    print("finished training",Config.max_num_generations, "genertations")
    generation.pareto_population.get_best_network()


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
    parser.add_argument('--protect', action='store_true', help='Protects from possible graph parsing errors')
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
        Config.device = args.device
        Config.num_workers = args.n_workers
        Config.max_num_generations = args.ngen
        Config.num_gpus = args.n_gpus
        if len(args.second) == 2:
            Config.second_objective, second_obj_comp = args.second
        if len(args.second) == 2:
            Config.third_objective, third_obj_comp = args.third
        Config.dummy_run = args.fake
        Config.protect_parsing_from_errors = args.protect
        Config.save_best_graphs = args.graph_save

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
