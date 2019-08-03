import sys
import os

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_2 = os.path.split(dir_path)[0]
sys.path.append(dir_path_1)
sys.path.append(dir_path_2)

from src.EvolutionEnvironment.Generation import Generation
from src.Config.Config import Config, load
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
    # TODO this doesn't work when spawning new processes
    # load('../Config/config.ini')
    # parse_args()

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
    if generation.generation_number == -1:
        print('Starting run for:', Config.run_name)
        start_gen = 0
    else:
        print('Continuing run:', Config.run_name, 'at generation', generation)
        # Generations save after completing their step. before incrementing their generation number
        start_gen = generation.generation_number + 1

    start_time = time.time()

    if start_gen < Config.max_num_generations:
        for i in range(start_gen, Config.max_num_generations):
            print('Running gen', i)
            gen_start_time = time.time()
            # current_generation.evaluate(i)
            generation.evaluate(i)
            generation.step()
            print('Completed gen', i, '\ntotal elapsed time:', (time.time() - start_time), end="\n\n")

    print('Finished training', Config.max_num_generations, 'generations')
    generation.pareto_population.get_best_network()


def parse_args():
    parser = argparse.ArgumentParser(description='Runs the CoDeepNEAT algorithm')

    parser.add_argument('-p', '--data-path', type=str, nargs='?', default=Config.data_path,
                        help='Directory to store the training and test data')
    parser.add_argument('--dataset', type=str, nargs='?', default=Config.dataset,
                        choices=['mnist', 'fashion_mnist', 'cifar10'], help='Dataset to train with')
    parser.add_argument('-d', '--device', type=str, nargs='?', default=Config.device, choices=['cpu', 'gpu'],
                        help='Device to train on')
    parser.add_argument('--n-gpus', type=int, nargs='?', default=Config.num_gpus,
                        help='The number of GPUs available, make sure that --device is not cpu or leave it blank')
    parser.add_argument('-n', '--ngen', type=int, nargs='?', default=Config.max_num_generations,
                        help='Max number of generations to run CoDeepNEAT')
    parser.add_argument('-s', '--second', type=str, nargs='*', default=(Config.second_objective, 'lt'),
                        help='Second objective name and lt or gt to indicate if a lower or higher value is better')
    parser.add_argument('-t', '--third', type=str, nargs='*', default=(Config.third_objective, 'lt'),
                        help='Third objective name and lt or gt to indicate if a lower or higher value is better')
    parser.add_argument('-f', '--fake', action='store_true', default=Config.dummy_run,
                        help='Runs a dummy version, for testing')
    parser.add_argument('--protect', action='store_true', default=Config.protect_parsing_from_errors,
                        help='Protects from possible graph parsing errors')
    parser.add_argument('-g', '--graph-save', action='store_true', default=Config.save_best_graphs,
                        help='Saves the best graphs in a generation')

    args = parser.parse_args()

    print('Using args:', args)

    if args.second is not None and len(args.second) not in (0, 2):
        parser.error('Either give no values for second, or two, not {}.'.format(len(args.second)))

    if args.third is not None and len(args.third) not in (0, 2):
        parser.error('Either give no values for third, or two, not {}.'.format(len(args.third)))

    Config.data_path = args.data_path
    Config.dataset = args.dataset
    Config.device = args.device
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
