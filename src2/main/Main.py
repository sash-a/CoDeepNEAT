from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from typing import TYPE_CHECKING

import torch
import wandb

# For importing project files

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'src2'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'runs'))

from runs import RunsManager

from src2.Configuration import config
from src2.main.Generation import Generation
import src2.main.Singleton as Singleton
from src2.Phenotype.NeuralNetwork.Evaluator import FullTraining

if TYPE_CHECKING:
    pass


def main():
    # seeding and unseeding pytorch so that the 'random split' is deterministic
    # TODO is this the best way to enforce a deterministic split? Do we want torch to be seeded?
    torch.manual_seed(0)
    print("before parse run name: ", config.run_name)
    arg_parse()
    print("after arg name: ", config.run_name)
    _force_cuda_device_init()
    init_operators()
    generation = init_generation()
    init_wandb(generation.generation_number)

    print(config.__dict__)

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)
        generation.step_evaluation()
        RunsManager.save_generation(generation, config.run_name)
        generation.step_evolution()


def fully_train(n=1):
    arg_parse()
    RunsManager.load_config(run_name=config.run_name)
    FullTraining.fully_train_best_evolved_networks(config.run_name, n)


def arg_parse():
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--configs', nargs='+', type=str,
                        help='Path to all config files that will be used. (Earlier configs are given preference)',
                        required=False)
    args = parser.parse_args()
    # Reading configs in reverse order so that initial config files overwrite others
    if args.configs is None:
        return

    for cfg_file in reversed(args.configs):
        config.read(cfg_file)


def init_generation() -> Generation:
    print('args parsed', config.run_name)

    if not RunsManager.does_run_folder_exist(config.run_name):
        """"fresh run"""
        RunsManager.set_up_run_folder(config.run_name)
        RunsManager.save_config(config.run_name, config)
        generation: Generation = Generation()
        Singleton.instance = generation

    else:
        """continuing run"""
        RunsManager.load_config(config.run_name)
        generation: Generation = RunsManager.load_latest_generation(config.run_name)
        Singleton.instance = generation
        generation.step_evolution()

    return generation


def init_wandb(gen_num: int):
    if config.use_wandb:
        if gen_num == 0:  # this is the first generation, need to initialize wandb
            config.wandb_run_id = config.run_name + str(datetime.date.today()) + '_' + str(random.randint(1E5, 1E6))

            tags = config.wandb_tags
            if config.dummy_run:
                tags += ['TEST_RUN']

            wandb.init(project='cdn', name=config.run_name, tags=tags, dir='../../results', id=config.wandb_run_id)
            for key, val in config.__dict__.items():
                wandb.config[key] = val

            # need to re-add the new wandb_run_id into the saved config
            RunsManager.save_config(config.run_name, config)

        else:  # this is not the first generation, need to resume wandb
            print('trying to resume', config.wandb_run_id)
            wandb.init(project='cdn', resume=config.wandb_run_id)


def init_operators():
    from src2.Genotype.NEAT.Operators.PopulationRankers.SingleObjectiveRank import SingleObjectiveRank
    from src2.Genotype.NEAT.Operators.RepresentativeSelectors.BestRepSelector import BestRepSelector
    from src2.Genotype.NEAT.Operators.RepresentativeSelectors.CentroidRepSelector import CentroidRepSelector
    from src2.Genotype.NEAT.Operators.RepresentativeSelectors.RandomRepSelector import RandomRepSelector
    from src2.Genotype.NEAT.Operators.ParentSelectors.RouletteSelector import RouletteSelector
    from src2.Genotype.NEAT.Operators.ParentSelectors.TournamentSelector import TournamentSelector
    from src2.Genotype.NEAT.Operators.ParentSelectors.UniformSelector import UniformSelector
    from src2.Genotype.NEAT.Species import Species
    from src2.Genotype.NEAT.Population import Population

    if not config.multiobjective:
        Population.ranker = SingleObjectiveRank()
    else:
        # TODO multiobjective rank
        raise NotImplemented('Multi-objectivity is not yet implemented')

    if config.parent_selector.lower() == "uniform":
        Species.selector = UniformSelector()
    elif config.parent_selector.lower() == "roulette":
        Species.selector = RouletteSelector()
    elif config.parent_selector.lower() == "tournament":
        Species.selector = TournamentSelector(5)
    else:
        raise Exception("unrecognised parent selector in config: " + str(config.parent_selector).lower() +
                        " expected either: uniform | roulette | tournament")

    if config.representative_selector.lower() == "centroid":
        Species.representative_selector = CentroidRepSelector()
    elif config.representative_selector.lower() == "random":
        Species.representative_selector = RandomRepSelector()
    elif config.representative_selector.lower() == "best":
        Species.representative_selector = BestRepSelector()
    else:
        raise Exception("unrecognised representative selector in config: " + config.representative_selector.lower()
                        + " expected either: centroid | random | best")


def _force_cuda_device_init():
    """Needed because of a bug in pytorch/cuda: https://github.com/pytorch/pytorch/issues/16559"""
    for i in range(config.n_gpus):
        with torch.cuda.device(i):
            torch.tensor([1.]).cuda()


if __name__ == '__main__':
    main()
    # fully_train(n=1)
