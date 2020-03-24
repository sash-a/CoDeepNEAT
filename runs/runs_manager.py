from __future__ import annotations

import wandb
import inspect
import json
import os
import pickle
import shutil
from os.path import join, exists, dirname, abspath
from typing import TYPE_CHECKING

from configuration import config, internal_config

if TYPE_CHECKING:
    from src.main.generation import Generation


def load_latest_generation(run_name):
    latest = get_latest_generation(run_name)
    if latest < 0:
        shutil.rmtree(get_run_folder_path(run_name))
        print('Run folder exists, but no generation.pickle file. '
              'Run may not have completed generation 0 - deleting '
              'run folder and restarting with original config')
        return None

    return load_generation(latest, run_name)


def load_generation(generation_number, run_name):
    file_name = get_generation_file_path(generation_number, run_name)

    print('loading generation', file_name)
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
    internal_config.load(run_name)
    return config


def save_config(run_name, conf=config, config_name="config", use_wandb_override=True):
    """Saves config locally and uploads it to wandb if config.use_wandb is true"""
    from src.utils.wandb_utils import upload_config
    file_path = join(get_run_folder_path(run_name), config_name + '.json')

    with open(file_path, 'w+') as f:
        json.dump(conf.__dict__, f, indent=2)

    if conf.use_wandb and use_wandb_override:
        upload_config()
        wandb.save(file_path)


def _get_generation_name(generation_number):
    return "generation_" + str(generation_number) + ".pickle"


def get_generation_file_path(generation_number, run_name):
    return join(get_generations_folder_path(run_name), _get_generation_name(generation_number))


def get_latest_generation(run_name):
    latest = 0
    while exists(get_generation_file_path(latest, run_name)):
        latest += 1
    return latest - 1


def set_up_run_folder(run_name, wandb_save=True):
    if not run_folder_exists(run_name):
        os.makedirs(get_run_folder_path(run_name))
        os.makedirs(get_graphs_folder_path(run_name))
        os.makedirs(get_generations_folder_path(run_name))

        internal_config.save(run_name, wandb_save)


def get_graphs_folder_path(run_name):
    return join(get_run_folder_path(run_name), "graphs")


def get_fully_train_folder_path(run_name):
    return join(get_run_folder_path(run_name), 'fully_trained_models')


def get_generations_folder_path(run_name):
    return join(get_run_folder_path(run_name), "generations")


def get_run_folder_path(run_name):
    return join(__get_runs_folder_path(), run_name)


def run_folder_exists(run_name) -> bool:
    return exists(get_run_folder_path(run_name))


def __get_runs_folder_path():
    return dirname(abspath(inspect.getfile(inspect.currentframe())))
