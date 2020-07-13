import json
import os
from typing import Dict, Tuple
import runs.runs_manager as run_man
import configuration as cfg
from configuration import Config


def get_config_path(path: str, scheduler_run_name: str) -> Tuple[str, str]:
    """picks which config to use next."""
    config_paths = read_json(Config.build_file_path(path))  # dict: path -> num_runs

    for config_path in config_paths:
        n_runs = config_paths[config_path]
        for i in range(n_runs):
            config_dict = read_json(Config.build_file_path(config_path))
            scheduled_run_name = config_dict['run_name']
            run_name = _get_effective_run_name(scheduled_run_name, i, scheduler_run_name)
            run_path = run_man.get_run_folder_path(run_name)

            run_folder_exists = os.path.exists(run_path)
            if run_folder_exists:
                cfg.internal_config.load(run_name)

            run_currently_running_in_another_process = run_folder_exists and cfg.internal_config.running

            if run_currently_running_in_another_process:
                print('run {} is being run in another process, moving on'.format(run_name))
            if cfg.internal_config.finished or run_currently_running_in_another_process:
                cfg.internal_config.__init__()  # reset internal config
                continue

            print('scheduler running', run_name)

            if run_folder_exists:
                cfg.internal_config.running = True
                cfg.internal_config.save(run_name, False)

            return config_path, run_name

    raise Exception('Could not find any non-running/non-finished configs in the batch run')


def _get_effective_run_name(scheduled_run_base_name, run_number, scheduler_prefix):
    if scheduler_prefix and scheduler_prefix not in scheduled_run_base_name:
        # if the scheduler provides a name prefix - add it
        # if the schedulers name is redundant - don't add it
        scheduled_run_base_name = scheduler_prefix + '_' + scheduled_run_base_name

    return scheduled_run_base_name + "_" + repr(run_number)


def get_fully_train_state(run_name):
    """reads the inner config of the given run, and determines if it is to be evolved/ FT'd"""
    run_path = run_man.get_run_folder_path(run_name)

    path_exists = os.path.exists(run_path)
    if path_exists:
        cfg.internal_config.load(run_name)

    fully_training = cfg.internal_config.state == 'ft'
    continue_fully_training = fully_training and cfg.internal_config.ft_started
    cfg.internal_config.__init__()

    return fully_training, continue_fully_training


def read_json(path: str) -> Dict[str, any]:
    if '.json' not in path:
        path += '.json'
    with open(path) as f:
        return json.load(f)


def write_json(d: dict, path: str):
    with open(path, 'w+') as f:
        json.dump(d, f, indent=2)

# get_config_path('batch_configs/run_schedule.json')
