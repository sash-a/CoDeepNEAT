from __future__ import annotations

import inspect
import json
import os
import pickle
from os.path import join, exists, dirname, abspath
from typing import TYPE_CHECKING

from src2.Configuration import config

if TYPE_CHECKING:
    from src2.main.Generation import Generation


def load_latest_generation(run_name):
    latest = get_latest_generation(run_name)
    if latest < 0:
        raise FileNotFoundError('Run folder exists, but no generation.pickle file. '
                                'Run may not have completed generation 0 - delete the whole folder')
    return load_generation(latest, run_name)


def load_generation(generation_number, run_name):
    file_name = get_generation_file_path(generation_number, run_name)

    print('loading', file_name)
    pickle_in = open(file_name, "rb")
    try:
        gen = pickle.load(pickle_in)
    except Exception as e:
        print('failed to load', file_name)
        print('with error:\n', repr(e))
        return None

    pickle_in.close()
    return gen


def save_generation(generation: Generation, run_name):
    file_name = get_generation_file_path(generation.generation_number, run_name)
    pickle_out = open(file_name, "wb")
    pickle.dump(generation, pickle_out)
    pickle_out.close()


def load_config(run_name, config_name="config"):
    file_path = join(get_run_folder_path(run_name), config_name + '.json')
    config.read(file_path)


def save_config(run_name, conf=config,config_name="config"):
    file_path = join(get_run_folder_path(run_name), config_name + '.json')
    with open(file_path, 'w+') as f:
        json.dump(conf.__dict__, f)


def _get_generation_name(generation_number):
    return "generation_" + str(generation_number) + ".pickle"


def get_generation_file_path(generation_number, run_name):
    return join(get_generations_folder_path(run_name), _get_generation_name(generation_number))


def get_latest_generation(run_name):
    latest = 0
    while exists(get_generation_file_path(latest, run_name)):
        latest += 1
    return latest - 1


def set_up_run_folder(run_name):
    if not does_run_folder_exist(run_name):
        os.makedirs(get_run_folder_path(run_name))
        os.makedirs(get_graphs_folder_path(run_name))
        os.makedirs(get_generations_folder_path(run_name))


def get_graphs_folder_path(run_name):
    return join(get_run_folder_path(run_name), "graphs")


def get_generations_folder_path(run_name):
    return join(get_run_folder_path(run_name), "generations")


def get_run_folder_path(run_name):
    return join(get_runs_folder_path(), run_name)


def does_run_folder_exist(run_name) -> bool:
    return exists(get_run_folder_path(run_name))


def get_runs_folder_path():
    return dirname(abspath(inspect.getfile(inspect.currentframe())))
