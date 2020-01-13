from __future__ import annotations

import argparse
import os
import sys
import torch

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
from src2.configuration import config
from src2.main.generation import Generation
from src2.phenotype.neural_network.evaluator import fully_train
from src2.utils.wandb_utils import init_wandb, wandb_log
from src2.phenotype.neural_network.evaluator.fully_train import fully_train


def main():
    read_config()
    if config.device == 'gpu':
        _force_cuda_device_init()

    locally_new_run = not runs_manager.does_run_folder_exist(config.run_name)
    downloading_run = bool(config.wandb_run_id)

    if config.fully_train:
        init_fully_train(locally_new_run)
        fully_train(config.run_name, epochs=config.fully_train_epochs)  # fully train evaluation
        return

    init_generation_dir(locally_new_run)  # A config from a previous run has possibly been loaded from this point

    init_wandb(locally_new_run)
    read_config()  # allowing provided config to overwrite downloaded one
    runs_manager.save_config(config.run_name, config)

    init_operators()
    generation = init_generation(False if downloading_run else locally_new_run)

    print('config:', config.__dict__)

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)
        generation.step_evaluation()

        runs_manager.save_generation(generation, config.run_name)
        if config.use_wandb:
            wandb_log(generation)

        generation.step_evolution()


def init_fully_train(locally_new_run: bool):
    print('Starting fully training...')

    downloading_conf = bool(config.wandb_run_id)
    init_wandb(locally_new_run)  # possibly downloading remote config

    if downloading_conf:
        # Downloaded a config, need to overwrite some of its values with values specified in local config

        # Values which may not be overwritten:
        remote_config_name = config.run_name
        ft_wandb_run_id = config.wandb_run_id

        read_config()  # overwrites some of downloaded config values with config that is passed as arg

        config.run_name = remote_config_name  # files are downloaded to a dir with this name so must continue to use it
        if not config.resume_fully_train:  # wandb run ID of fully train gets overwritten by init config with the evo ID
            config.wandb_run_id = ft_wandb_run_id

    runs_manager.save_config(config.run_name, config)


def read_config():
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


def init_generation_dir(new_run: bool):
    if new_run:
        runs_manager.set_up_run_folder(config.run_name)
    else:
        runs_manager.load_config(config.run_name)


def init_generation(new_run: bool) -> Generation:
    if new_run:
        # new run
        generation: Generation = Generation()
        Singleton.instance = generation

    else:
        # continuing run
        generation: Generation = runs_manager.load_latest_generation(config.run_name)
        if generation is None:  # generation load failed, likely because the run did not complete gen 0
            init_generation_dir(True)
            return init_generation(True)  # will start a fresh gen

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
