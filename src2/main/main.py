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

import src2.main.singleton as Singleton

from runs import runs_manager
from src2.analysis.datafetcher import fetch_generations, fetch_config
from src2.configuration import config
from src2.main.generation import Generation
from src2.phenotype.neural_network.evaluator import fully_train

if TYPE_CHECKING:
    pass


def main():
    locally_new_run = not runs_manager.does_run_folder_exist(config.run_name)  # this must come before config is created
    init_config(locally_new_run)
    downloading_run = bool(config.wandb_run_id)  # this must come after config is written to

    init_wandb(locally_new_run)
    init_operators()
    print(downloading_run, locally_new_run)
    print(False if downloading_run else locally_new_run)
    generation = init_generation(False if downloading_run else locally_new_run)

    if config.device == 'gpu':
        _force_cuda_device_init()

    if config.fully_train:
        init_wandb(locally_new_run)
        fully_train()
        return

    print('config:', config.__dict__)

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)
        generation.step_evaluation()
        runs_manager.save_generation(generation, config.run_name)
        generation.step_evolution()


def fully_train(n=1):
    init_config(False)
    runs_manager.load_config(run_name=config.run_name)
    fully_train.fully_train(config.run_name, n)


def init_config(new_run: bool):
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

    if new_run:
        runs_manager.set_up_run_folder(config.run_name)
        runs_manager.save_config(config.run_name, config)
    else:
        runs_manager.load_config(config.run_name)


def init_generation(new_run: bool) -> Generation:
    if new_run:
        print('new')
        # new run
        generation: Generation = Generation()
        Singleton.instance = generation

    else:
        # continuing run
        print('cont')
        generation: Generation = runs_manager.load_latest_generation(config.run_name)
        if generation is None:  # generation load failed, likely because the run did not complete gen 0
            return init_generation(True)  # will start a fresh gen

        Singleton.instance = generation
        generation.step_evolution()

    return generation


def init_wandb(is_new_run):
    if not config.use_wandb:
        return

    print(config.wandb_run_id, bool(config.wandb_run_id), is_new_run)
    if config.wandb_run_id and is_new_run:
        # the runs folder does not exist locally, but a run ID is provided -> create the folder and download the run
        print('fetching!!!')
        fetch_generations(run_id=config.wandb_run_id)
        # config used to download the run will already be copied there so must replace it
        fetch_config(run_id=config.wandb_run_id, replace=True)
        wandb.init(project='cdn', entity='codeepneat', resume=config.wandb_run_id)
    elif is_new_run or config.fully_train:
        # this is the first generation -> initialize wandb
        config.wandb_run_id = config.run_name + str(datetime.date.today()) + '_' + str(random.randint(1E5, 1E6))

        project = 'cdn_fully_train' if config.fully_train else 'cdn'
        dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')
        tags = config.wandb_tags
        if config.fully_train:
            tags = ['FULLY_TRAIN']
        if config.dummy_run:
            tags += ['TEST_RUN']

        wandb.init(project=project, entity='codeepneat', name=config.run_name, tags=tags, dir=dir,
                   id=config.wandb_run_id)
        wandb.config.update(config.__dict__)

        # need to re-add the new wandb_run_id into the saved config
        runs_manager.save_config(config.run_name, config)
        # saving the config to wandb
        wandb.save(os.path.join(runs_manager.get_run_folder_path(config.run_name), 'config.json'))
    else:
        # this is not the first generation -> need to resume wandb
        print('attempting to resume', config.wandb_run_id)
        wandb.init(project='cdn', entity='codeepneat', resume=config.wandb_run_id)


def init_operators():
    from src2.genotype.neat.operators.population_rankers.single_objective_rank import SingleObjectiveRank
    from src2.genotype.neat.operators.representative_selectors.best_rep_selector import BestRepSelector
    from src2.genotype.neat.operators.representative_selectors.centroid_rep_selector import CentroidRepSelector
    from src2.genotype.neat.operators.representative_selectors.random_rep_selector import RandomRepSelector
    from src2.genotype.neat.operators.parent_selectors.roulette_selector import RouletteSelector
    from src2.genotype.neat.operators.parent_selectors.tournament_selector import TournamentSelector
    from src2.genotype.neat.operators.parent_selectors.uniform_selector import UniformSelector
    from src2.genotype.neat.species import Species
    from src2.genotype.neat.population import Population

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
        try:
            with torch.cuda.device(i):
                torch.tensor([1.]).cuda()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
