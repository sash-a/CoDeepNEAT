import os
import time

import torch

import src.main.singleton as Singleton
from runs import runs_manager as run_man
from src.main.generation import Generation
from configuration import config, internal_config
from src.utils.wandb_utils import wandb_log


def _force_cuda_device_init():
    """Needed because of a bug in pytorch/cuda: https://github.com/pytorch/pytorch/issues/16559"""
    if config.device != 'gpu':
        return

    for i in range(config.n_gpus):
        try:
            with torch.cuda.device(i):
                torch.tensor([1.]).cuda()
        except Exception as e:
            print(e)


def init_generation() -> Generation:
    """either loads a saved gen from a to-be-continued run, or creates a fresh one"""
    if not os.listdir(run_man.get_generations_folder_path(config.run_name)):
        # if there are no generations in the run folder
        generation: Generation = Generation()
        Singleton.instance = generation
    else:  # continuing run
        generation: Generation = run_man.load_latest_generation(config.run_name)
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


def timeit(func):
    """
    Decorator that returns the total runtime of a function
    @param func: function to be timed
    @return: (func, time_taken). Time is in seconds
    """

    def wrapper(*args, **kwargs) -> float:
        start = time.time()
        func(*args, **kwargs)
        total_time = time.time() - start
        return total_time

    return wrapper


@timeit
def step_generation(generation: Generation):
    generation.step_evaluation()

    run_man.save_generation(generation, config.run_name)
    if config.use_wandb:
        wandb_log(generation)

    generation.step_evolution()


def evolve():
    print('Evolving')
    allowed_gen_time = -1
    run_time = 0
    end_time = config.allowed_runtime_sec

    init_operators()
    generation = init_generation()

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)

        # Checks if generation is expected to run allowed time
        if config.allowed_runtime_sec != -1 and end_time - run_time < allowed_gen_time:
            print('Stopped run (gen {}) because the next generation will likely to take longer than the remaining time'
                  .format(generation.generation_number))
            return

        gen_time = step_generation(generation)
        allowed_gen_time = max(allowed_gen_time, gen_time * 1.05)
        run_time += gen_time

        print(f'runtime: {run_time}, end time: {end_time}, allowing time {allowed_gen_time}')

    internal_config.state = 'ft'
