import json
import os
from typing import Dict, Tuple
import runs.runs_manager as run_man
import configuration as cfg

BATCH_RUNS = 3


def build_file_path(file: str) -> str:
    # If the path is not absolute (i.e starts at root) then search in configs dir
    if not file.endswith('.json'):
        file += ".json"

    if not file.startswith("/"):
        file = os.path.join(os.path.dirname(__file__), 'configs', file)

    return file


def get_config_path(path: str) -> Tuple[str, str]:
    config_paths = read_json(path)  # dict: path -> num_runs

    for config_path in config_paths:
        for i in range(BATCH_RUNS):
            config_dict = read_json(build_file_path(config_path))
            run_name = config_dict['run_name'] + str(i)
            run_path = run_man.get_run_folder_path(run_name)

            path_exists = os.path.exists(run_path)
            if path_exists:
                cfg.internal_config.load(run_name)

            if cfg.internal_config.finished or path_exists and cfg.internal_config.running:
                cfg.internal_config.__init__()
                continue

            if cfg.internal_config.state == 'ft':
                # make config options ft (path must exist at this point)
                cfg.config.read(os.path.join(run_path, 'config.json'))
                cfg.config.fully_train = True
                cfg.config.resume_fully_train = cfg.internal_config.ft_epoch > 0
                run_man.save_config(run_name, cfg.config)

            # incrementing the number of runs
            config_paths[config_path] += 1
            write_json(config_paths, path)

            return config_path, str(i)

    raise Exception('Could not find any non-running/non-finished configs in the batch run')


def read_json(path: str) -> Dict[str, any]:
    with open(path) as f:
        return json.load(f)


def write_json(d: dict, path: str):
    with open(path, 'w+') as f:
        json.dump(d, f, indent=2)

# get_config_path('batch_configs/run_schedule.json')
