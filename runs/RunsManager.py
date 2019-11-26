from __future__ import annotations

import inspect
import os
import pickle
from os.path import join, exists, dirname, abspath
from typing import TYPE_CHECKING

from src2.Configuration import config

if TYPE_CHECKING:
    from src2.main.Generation import Generation


def load_latest_generation(run_name=config.run_name):
    latest_generation = _get_latest_generation(run_name)
    file_name = _get_generation_file_path(latest_generation, run_name)
    if latest_generation <0:
        raise Exception("run folder, but no generation pickle. run may not have completed gen 0 - delete it")
    print("loading", file_name)
    pickle_in = open(file_name, "rb")
    try:
        gen = pickle.load(pickle_in)
    except:
        print("failed to load",file_name)
        return None

    pickle_in.close()
    return gen


def save_generation(generation: Generation, run_name=config.run_name):
    file_name = _get_generation_file_path(generation.generation_number, run_name)
    pickle_out = open(file_name, "wb")
    pickle.dump(generation, pickle_out)
    pickle_out.close()


def _get_generation_name(generation_number):
    return "generation_" + str(generation_number) + ".pickle"


def _get_generation_file_path(generation_number, run_name):
    return join(get_generations_folder_path(run_name), _get_generation_name(generation_number))


def _get_latest_generation(run_name):
    latest = 0
    while exists(_get_generation_file_path(latest, run_name)):
        latest += 1
    return latest - 1


def set_up_run_folder(run_name=config.run_name):
    if not does_run_folder_exist(run_name):
        os.makedirs(get_run_folder_path(run_name))
        os.makedirs(get_graphs_folder_path(run_name))
        os.makedirs(get_generations_folder_path(run_name))


def get_graphs_folder_path(run_name=config.run_name):
    return join(get_run_folder_path(run_name), "graphs")


def get_generations_folder_path(run_name=config.run_name):
    return join(get_run_folder_path(run_name), "generations")


def get_run_folder_path(run_name=config.run_name):
    return join(get_runs_folder_path(), run_name)


def does_run_folder_exist(run_name=config.run_name):
    return exists(get_run_folder_path(run_name))


def get_runs_folder_path():
    return dirname(abspath(inspect.getfile(inspect.currentframe())))
