import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'runs'))
sys.path.append(os.path.join(dir_path_1, 'configuration'))

import json
from runs.runs_manager import get_fully_train_folder_path, get_run_folder_path, __get_runs_folder_path


def wipe(run_name: str):
    saved_models_dir = get_fully_train_folder_path(run_name)
    for file in os.listdir(saved_models_dir):
        path = os.path.join(saved_models_dir, file)
        print(f'deleting saved model at: {path}')
        os.remove(path)


def reset_internal_config(run_name: str):
    j = json.load(open(os.path.join(get_run_folder_path(run_name), 'internal_config.json'), 'r'))

    if j['state'] == 'finished':
        j['state'] = 'ft'
        j['ft_started'] = False

    j['finished'] = False

    # j['running'] = False

    json.dump(j, open(os.path.join(get_run_folder_path(run_name), 'internal_config.json'), 'w'))


if __name__ == '__main__':
    dirs = [d for d in os.listdir(__get_runs_folder_path()) if os.path.isdir(get_run_folder_path(d))]
    for dir in dirs:
        try:
            wipe(dir)
            reset_internal_config(dir)
        except FileNotFoundError:
            print(f'no relevant files in {dir}')
