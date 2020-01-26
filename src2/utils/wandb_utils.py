from __future__ import annotations

import os
import datetime
from random import randint
from typing import TYPE_CHECKING, List, Dict
from PIL import Image

import re
import wandb

from runs.runs_manager import get_generation_file_path, save_config, get_run_folder_path, get_graphs_folder_path, \
    run_folder_exists
from src2.configuration import config
from src2.utils.wandb_data_fetcher import download_generations, download_config, download_model

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
        # Resuming a remote fully train
        _fetch_run()
        _resume_run(True)
        download_generations(run_path='codeepneat/cdn/' + wandb.config.evolution_run_id, replace=True)

    elif config.wandb_run_id and config.fully_train and not config.resume_fully_train:
        # Start a fully train given remote evolution run
        evolution_run_id = config.wandb_run_id  # ID of the evolution run, because its before we download the new config
        _fetch_run()  # fetches a remote fun
        _new_run(True)  # Then creates a new fully train run
        # this is done to allow fully train runs to resume without needing the local generation files for that run
        wandb.config['evolution_run_id'] = evolution_run_id

    elif config.wandb_run_id and not config.fully_train:
        # Resume remote evolution run
        _fetch_run()  # Fetches remote run
        _resume_run(False)  # Loads that run

    elif is_new_run or config.fully_train:
        # start new local run
        _new_run(config.fully_train)

    else:
        # resume local run
        _resume_run(config.fully_train)


def wandb_init():
    if (config.fully_train and not config.resume_fully_train) or \
            (not run_folder_exists(config.run_name) and not config.wandb_run_path):
        # Either new evolution run or new fully train run
        evo_run_path = config.wandb_run_path
        print('new run')
        _new_run()
        if config.fully_train:
            wandb.config['evolution_run_path'] = evo_run_path

    elif run_folder_exists(config.run_name) or config.resume_fully_train:
        print('resuming')
        _resume_run()
        if config.resume_fully_train:
            print('resuming ft at', wandb.config.evolution_run_path)
            download_generations(run_path=wandb.config.evolution_run_path, replace=True)
            download_model(run_path=wandb.config.evolution_run_path, replace=True)
    else:
        raise Exception("Something went wrong with wandb")

    wandb.config.update(config.__dict__, allow_val_change=True)


def _fetch_run():
    if config.resume_fully_train:
        path_prefix = 'codeepneat/cdn_fully_train'
        download_model(run_path=path_prefix + '/' + config.wandb_run_id, replace=True)
    else:
        path_prefix = 'codeepneat/cdn'
        download_generations(run_path=path_prefix + '/' + config.wandb_run_id, replace=True)

    # config used to download the run will already be copied there so must replace it
    download_config(run_path=path_prefix + '/' + config.wandb_run_id, replace=True)


def _resume_run():
    project = 'cdn_fully_train' if config.fully_train else 'cdn'
    dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')
    print('RPPRPRPRPPRR',config.wandb_run_path.split('/'),config.wandb_run_path)
    wandb.init(dir=dir, project=project, entity='codeepneat', resume=config.wandb_run_path.split('/')[2])


def _new_run():
    wandb_run_id = config.run_name + str(datetime.date.today()) + '_' + str(randint(1E5, 1E6))
    project = 'cdn_fully_train' if config.fully_train else 'cdn'
    config.wandb_run_path = 'codeepneat/' + project + '/' + wandb_run_id

    dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')

    job_type = 'train'
    tags = config.wandb_tags
    if config.fully_train:
        tags = ['FULLY_TRAIN']
    if config.dummy_run:
        tags += ['TEST_RUN']
        job_type = 'test'

    wandb.init(job_type=job_type, project=project, entity='codeepneat', name=config.run_name, tags=tags, dir=dir,
               id=wandb_run_id)


def wandb_log(generation: Generation):
    if not config.use_wandb:
        return
    _wandb_log_generation(generation)


def _wandb_log_generation(generation: Generation):
    module_accs = sorted([module.accuracy for module in generation.module_population])
    bp_accs = sorted([bp.accuracy for bp in generation.blueprint_population])

    n_unevaluated_bps = 0
    n_large_nets = 0
    raw_bp_accs = []
    for bp in generation.blueprint_population:
        evaluated_networks = len([fitness[0] for fitness in bp.fitness_raw if fitness[0] != 0])
        n_unevaluated_bps += 1 if evaluated_networks == 0 else 0
        n_large_nets += sum(fitness[0] == 0 for fitness in bp.fitness_raw)
        raw_bp_accs.extend(bp.fitness_raw[0])

    n_unevaluated_mods = 0
    raw_mod_accs = []
    for mod in generation.module_population:
        n_unevaluated_mods += 1 if mod.n_evaluations == 0 else 0
        raw_mod_accs.extend(mod.fitness_raw[0])  # this will end up being a list in MOO

    mod_acc_tbl = wandb.Table(['module accuracies'], data=raw_mod_accs)
    bp_acc_tbl = wandb.Table(['blueprint accuracies'], data=raw_bp_accs)

    non_zero_mod_accs = [x for x in raw_mod_accs if x != 0]

    # Saving the pickle file for further inspection
    # Were getting some runs that did not upload all generation files so now re-save all every generation to make sure
    for i in range(generation.generation_number + 1):
        wandb.save(get_generation_file_path(i, config.run_name))

    log = {'module accuracy table': mod_acc_tbl, 'blueprint accuracy table': bp_acc_tbl,
           config.fitness_aggregation + ' module accuracies': module_accs,
           config.fitness_aggregation + ' blueprint accuracies': bp_accs,
           'module accuracies raw': raw_mod_accs, 'blueprint accuracies raw': raw_bp_accs,
           'avg module accuracy': sum(non_zero_mod_accs) / len(non_zero_mod_accs),
           'avg blueprint accuracy': sum(bp_accs) / len(bp_accs),
           'best blueprint accuracy': max(raw_bp_accs),
           'num module species': len(generation.module_population.species),
           'species sizes': [len(spc.members) for spc in generation.module_population.species],
           'unevaluated blueprints': n_unevaluated_bps, 'n_unevaluated_mods': n_unevaluated_mods,
           "large blueprints": n_large_nets, 'speciation threshold': generation.module_population.speciator.threshold,
           }

    if config.plot_best_genotypes or config.plot_best_phenotype:
        imgs = _log_imgs(generation)
        log.update(imgs)

    wandb.log(log)


def _log_imgs(generation: Generation) -> Dict[str, wandb.Image]:
    imgs = {}
    for root, _, files in os.walk(get_graphs_folder_path(config.run_name)):
        for file in files:
            if not file.endswith('.png'):
                continue

            for match in re.compile(r"_g[0-9]+_").finditer(str(file)):
                if match[0][2:-1] == str(generation.generation_number - 1):
                    name = 'best_pheno' if 'phenotype' in file else 'best_geno'
                    imgs[name] = wandb.Image(Image.open(os.path.join(root, file)), file)

    return imgs
