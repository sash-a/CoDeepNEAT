from __future__ import annotations

import atexit
import argparse
import os
import sys
from typing import Tuple

import torch

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'runs'))
sys.path.append(os.path.join(dir_path_1, 'configuration'))

import src.main.singleton as Singleton
from src.utils.wandb_data_fetcher import download_run
from runs import runs_manager
from configuration import config, internal_config, batch_runner
from src.main.generation import Generation
from src.phenotype.neural_network.evaluator import fully_train
from src.utils.wandb_utils import wandb_log, wandb_init
from src.phenotype.neural_network.evaluator.fully_train import fully_train


def main():
    run_name_suffix = ''
    cfg_file_path, n_gpus = get_args()
    batch_run_scheduler = config.read_option(cfg_file_path, 'batch_run_scheduler')

    if batch_run_scheduler:  # there is a batch run scheduler so must use the config specified in it
        cfg_file_path, run_name_suffix = batch_runner.get_config_path(batch_run_scheduler)

    wandb_run_path = config.read_option(cfg_file_path, 'wandb_run_path')
    run_name = config.read_option(cfg_file_path, 'run_name') + run_name_suffix

    if run_name is None or not run_name:
        raise Exception("Config file must have the run_name attribute")

    print('wandb run path:', wandb_run_path)
    if wandb_run_path is not None and wandb_run_path:
        print('downloading run...')
        run_name = download_run(run_path=wandb_run_path, replace=True)

    print('Run name', run_name)
    if runs_manager.run_folder_exists(run_name):
        print('Run folder already exists, reading its config')
        runs_manager.load_config(run_name)

    print('Reading config at ', cfg_file_path)
    config.read(cfg_file_path)  # overwrites loaded config with config passed as arg
    if run_name is not None:
        config.run_name = run_name  # if suffix has been added to run folder, then add it to config.run_name
    if n_gpus is not None:
        config.n_gpus = n_gpus
    else:
        print("no gpu argument given, using config value of", config.n_gpus)

    # Full config is now loaded
    if config.use_wandb:
        wandb_init()

    if not runs_manager.run_folder_exists(run_name):
        print('New run, setting up run folder')
        runs_manager.set_up_run_folder(run_name)

    print('Saving conf')
    print('config:', config.__dict__)
    runs_manager.save_config(config.run_name)

    if config.device == 'gpu':
        _force_cuda_device_init()

    if config.fully_train:
        fully_train(config.run_name, epochs=config.fully_train_epochs)
    else:
        evolve()


def get_args() -> Tuple[str, int]:
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--config', type=str, help='Config file that will be used', required=True)
    parser.add_argument('-g', '--ngpus', type=int, help='Number of GPUs available', required=False)

    args = parser.parse_args()
    return args.config, args.ngpus


def evolve():
    print('Evolving')
    init_operators()
    generation = init_generation()

    while internal_config.generation < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)
        generation.step_evaluation()

        runs_manager.save_generation(generation, config.run_name)
        if config.use_wandb:
            wandb_log(generation)

        generation.step_evolution()

    internal_config.state = 'ft'


def init_generation_dir(new_run: bool):
    if new_run:
        runs_manager.set_up_run_folder(config.run_name)
    else:
        runs_manager.load_config(config.run_name)


def init_generation() -> Generation:
    if not os.listdir(runs_manager.get_generations_folder_path(config.run_name)):  # new run
        generation: Generation = Generation()
        Singleton.instance = generation
    else:  # continuing run
        generation: Generation = runs_manager.load_latest_generation(config.run_name)
        if generation is None:  # generation load failed, likely because the run did not complete gen 0
            init_generation_dir(True)
            return init_generation()  # will start a fresh gen

        Singleton.instance = generation
        generation.step_evolution()

    return generation


def init_operators():
    from src.genotype.neat.operators.representative_selectors.best_rep_selector import BestRepSelector
    from src.genotype.neat.operators.representative_selectors.centroid_rep_selector import CentroidRepSelector
    from src.genotype.neat.operators.representative_selectors.random_rep_selector import RandomRepSelector
    from src.genotype.neat.operators.parent_selectors.roulette_selector import RouletteSelector
    from src.genotype.neat.operators.parent_selectors.tournament_selector import TournamentSelector
    from src.genotype.neat.operators.parent_selectors.uniform_selector import UniformSelector
    from src.genotype.neat.species import Species

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
    atexit.register(internal_config.on_exit)
    main()
