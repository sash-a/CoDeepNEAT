from __future__ import annotations

import os
import datetime
from random import randint
from typing import List, TYPE_CHECKING
import wandb

from runs.runs_manager import get_generation_file_path, save_config, get_run_folder_path
from src2.configuration import config
from src2.utils.wandb_data_fetcher import fetch_generations, fetch_config, fetch_model

if TYPE_CHECKING:
    from src2.main.generation import Generation
    from src2.phenotype.neural_network.neural_network import Network


def init_wandb(is_new_run):
    """
    Initializes wandb either as a new run or resuming an old run. If resuming from wandb then the wandb copy is
    preferred to the local one. So if the run already exists locally and a wandb ID is provided then the local run
    will be overwritten.
    """
    if not config.use_wandb:
        return

    if config.wandb_run_id and config.fully_train and config.resume_fully_train:
        # Resuming a checkpoint of a fully train
        _fetch_run()
    elif config.wandb_run_id and config.fully_train and not config.resume_fully_train:
        # Download a run
        _fetch_run()  # fetches a remote fun
        _new_run()  # Then creates a new fully train run
    elif config.wandb_run_id and not config.fully_train:
        # Run ID is provided and not fully train -> create the folder and download the run
        _fetch_run()  # Fetches remote run
        _load_local_run()  # Loads that run
    elif is_new_run or config.fully_train:
        # this is the first generation -> initialize wandb
        _new_run()
    else:
        # this is not the first generation -> need to resume wandb
        _load_local_run()


def _fetch_run():
    if config.resume_fully_train:
        fetch_model(replace=True)

    fetch_generations(run_id=config.wandb_run_id, replace=True)
    # config used to download the run will already be copied there so must replace it
    fetch_config(run_id=config.wandb_run_id, replace=True)


def _load_local_run():
    wandb.init(project='cdn', entity='codeepneat', resume=config.wandb_run_id)


def _new_run():
    config.wandb_run_id = config.run_name + str(datetime.date.today()) + '_' + str(randint(1E5, 1E6))

    project = 'cdn_fully_train' if config.fully_train else 'cdn'
    dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')
    tags = config.wandb_tags
    if config.fully_train:
        tags = ['FULLY_TRAIN']
    if config.dummy_run:
        tags += ['TEST_RUN']

    wandb.init(project=project, entity='codeepneat', name=config.run_name, tags=tags, dir=dir,
               id=config.wandb_run_id)
    wandb.config.update(config.__dict__)

    # need to re-add the new wandb_run_id into the saved config
    save_config(config.run_name, config)
    # saving the config to wandb
    wandb.save(os.path.join(get_run_folder_path(config.run_name), 'config.json'))


def wandb_log(generation: Generation):
    if not config.use_wandb:
        return
    _wandb_log_generation(generation)


def _wandb_log_generation(generation: Generation):
    module_accs = sorted([module.accuracy for module in generation.module_population])
    bp_accs = sorted([bp.accuracy for bp in generation.blueprint_population])

    n_unevaluated_bps = 0
    raw_bp_accs = []
    for bp in generation.blueprint_population:
        n_unevaluated_bps += sum(fitness[0] == 0 for fitness in bp.fitness_raw)
        raw_bp_accs.extend(bp.fitness_raw[0])

    n_unevaluated_mods = 0
    raw_mod_accs = []
    for mod in generation.module_population:
        n_unevaluated_mods += 1 if mod.n_evaluations == 0 else 0
        raw_mod_accs.extend(mod.fitness_raw[0])

    mod_acc_tbl = wandb.Table(['module accuracies'], data=raw_mod_accs)
    bp_acc_tbl = wandb.Table(['blueprint accuracies'], data=raw_bp_accs)
    # Saving the pickle file for further inspection
    wandb.save(get_generation_file_path(generation.generation_number, config.run_name))

    wandb.log({'module accuracy table': mod_acc_tbl, 'blueprint accuracy table': bp_acc_tbl,
               'module accuracies aggregated': module_accs, 'blueprint accuracies aggregated': bp_accs,
               'module accuracies raw': raw_mod_accs, 'blueprint accuracies raw': raw_bp_accs,
               'avg module accuracy': sum(module_accs) / len(module_accs),
               'avg blueprint accuracy': sum(bp_accs) / len(bp_accs),
               'best module accuracy': max(raw_mod_accs), 'best blueprint accuracy': max(raw_bp_accs),
               'num module species': len(generation.module_population.species),
               'species sizes': [len(spc.members) for spc in generation.module_population.species],
               'unevaluated blueprints': n_unevaluated_bps, 'n_unevaluated_mods': n_unevaluated_mods,
               'speciation threshold': generation.module_population.speciator.threshold
               })
