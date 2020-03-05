from __future__ import annotations

import argparse
import atexit
import os
import sys
from typing import Optional

# For importing project files
from src.utils.main_common import _force_cuda_device_init, init_operators, init_generation, step_generation

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'runs'))
sys.path.append(os.path.join(dir_path_1, 'configuration'))

from runs import runs_manager as run_man
from configuration import config, internal_config
from src.utils.wandb_utils import new_evo_run, resume_evo_run


def main():
    args = get_cli_args()
    load_config(args.config, args.ngpus)

    _force_cuda_device_init()

    evolve()


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--config', type=str, help='Config file that will be used', required=True)
    parser.add_argument('-g', '--ngpus', type=int, help='Number of GPUs available', required=False)

    return parser.parse_args()


def load_config(config_path: str, ngpus: Optional[int]):
    """
    Steps:
    * Read config to run name and wandb related info
    * Read saved config if it is a saved run
    * Load wandb if wandb is requested
    * Read original config again for overwrites
    * Overwrite n_gpus option if required
    * Save config to run folder

    @param config_path: path to the config, can be relative to configuration/configs
    @param ngpus: number of gpus if one should override config option
    """
    config.read(config_path)
    run_name = config.run_name

    print('Run name', run_name)
    if run_man.run_folder_exists(run_name):
        print('Run folder already exists, reading its config')
        run_man.load_config(run_name)  # load saved config
        if config.use_wandb:
            resume_evo_run()
    else:
        print('No runs folder detected with name {}. Creating one'.format(run_name))
        if config.use_wandb:
            new_evo_run()

        run_man.set_up_run_folder(config.run_name)

    config.read(config_path)  # overwrite with provided config

    if ngpus is not None:  # n_gpu override
        config.n_gpus = ngpus

    run_man.save_config(run_name)


def evolve():
    print('Evolving')
    gen_time = -1
    run_time = 0
    end_time = config.allowed_runtime_sec

    init_operators()
    generation = init_generation()

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)

        # Checks if generation is expected to run allowed time
        if config.allowed_runtime_sec != -1 and end_time - run_time < gen_time:
            print('Stopped run (gen {}) because the next generation will likely to take longer than the remaining time'
                  .format(generation.generation_number))
            return

        run_time = step_generation(generation)
        gen_time = max(gen_time, run_time * 1.15)

    internal_config.state = 'ft'


if __name__ == '__main__':
    atexit.register(internal_config.on_exit)
    main()
