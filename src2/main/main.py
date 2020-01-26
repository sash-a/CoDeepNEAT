from __future__ import annotations

import argparse
import os
import sys
import torch

# For importing project files
from src2.utils.wandb_data_fetcher import download_run

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'src2'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'runs'))

import src2.main.singleton as Singleton

from runs import runs_manager
from src2.configuration import config
from src2.main.generation import Generation
from src2.phenotype.neural_network.evaluator import fully_train
from src2.utils.wandb_utils import wandb_log, wandb_init
from src2.phenotype.neural_network.evaluator.fully_train import fully_train


def main():
    cfg_file_path = get_cfg_file_path()
    run_path = None if cfg_file_path is None else config.read_option(cfg_file_path, 'wandb_run_path')
    run_name = config.read_option(cfg_file_path, 'run_name')

    print('wandb run path:', run_path is None)
    if run_path is not None and run_path:
        print('downloading run...')
        run_name = download_run(run_path=run_path, replace=True)

    print('Run name', run_name)
    if runs_manager.run_folder_exists(run_name):
        print('Run folder already exists, reading its config')
        runs_manager.load_config(run_name)

    print('Reading config at ', cfg_file_path)
    config.read(cfg_file_path)  # overwrites loaded config with config passed as arg

    # Full config is now loaded
    if config.use_wandb:
        wandb_init()

    if not runs_manager.run_folder_exists(run_name):
        print('New run, setting up run folder')
        runs_manager.set_up_run_folder(run_name)

    print('Saving conf')
    runs_manager.save_config(config.run_name)

    if config.device == 'gpu':
        _force_cuda_device_init()

    if config.fully_train:
        fully_train(config.run_name)  # either load generations or load model and resume fully train
    else:
        evolve()


def get_cfg_file_path():
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--config', type=str,
                        help='Config file that will be used',
                        required=False)
    args = parser.parse_args()
    return args.config


def evolve():
    print('Evolving')
    init_operators()
    generation = init_generation()

    print('config:', config.__dict__)

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)
        generation.step_evaluation()

        runs_manager.save_generation(generation, config.run_name)
        if config.use_wandb:
            wandb_log(generation)

        generation.step_evolution()


def init_generation_dir(new_run: bool):
    if new_run:
        runs_manager.set_up_run_folder(config.run_name)
    else:
        runs_manager.load_config(config.run_name)


def init_generation() -> Generation:
    if not os.listdir(runs_manager.get_generations_folder_path(config.run_name)):  # new run
        print('new run')
        generation: Generation = Generation()
        Singleton.instance = generation
    else:  # continuing run
        print('cont run')
        generation: Generation = runs_manager.load_latest_generation(config.run_name)
        if generation is None:  # generation load failed, likely because the run did not complete gen 0
            init_generation_dir(True)
            return init_generation()  # will start a fresh gen

        Singleton.instance = generation
        generation.step_evolution()

    return generation


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
