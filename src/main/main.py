from __future__ import annotations

import atexit
import os
import sys
import time

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
from runs import runs_manager
from configuration import config, internal_config, config_loader
from src.main.generation import Generation
from src.phenotype.neural_network.evaluator import fully_train
from src.utils.wandb_utils import wandb_log, wandb_init
from src.phenotype.neural_network.evaluator.fully_train import fully_train


def main():
    config_loader.load_batch_config()

    if config.device == 'gpu':
        _force_cuda_device_init()

    if config.fully_train:
        fully_train(config.run_name)
    else:
        evolve()


def evolve():
    print('Evolving')
    gen_time = -1
    start_time = time.time()
    end_time = config.allowed_runtime_sec

    init_operators()
    generation = init_generation()

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)

        gen_start_time = time.time()
        run_time = time.time() - start_time
        if config.allowed_runtime_sec != -1 and end_time - run_time < gen_time:
            print('Stopped run (gen {}) because the next generation is supposed to take longer than the remaining time'
                  .format(generation.generation_number))
            return

        generation.step_evaluation()

        runs_manager.save_generation(generation, config.run_name)
        if config.use_wandb:
            wandb_log(generation)

        generation.step_evolution()
        gen_time = max(gen_time, (time.time() - gen_start_time) * 1.15)

    internal_config.state = 'ft'


def init_generation() -> Generation:
    """either loads a saved gen from a to-be-continued run, or creates a fresh one"""
    if not os.listdir(runs_manager.get_generations_folder_path(config.run_name)):
        # if there are no generations in the run folder
        generation: Generation = Generation()
        Singleton.instance = generation
    else:  # continuing run
        generation: Generation = runs_manager.load_latest_generation(config.run_name)
        if generation is None:  # generation load failed, likely because the run did not complete gen 0
            raise Exception("generation files exist, but failed to load")

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
