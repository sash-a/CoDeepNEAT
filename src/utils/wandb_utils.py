from __future__ import annotations

import os
import datetime
from random import randint
from typing import TYPE_CHECKING, Dict
from PIL import Image

import re
import wandb

from runs.runs_manager import get_generation_file_path, get_graphs_folder_path, \
    run_folder_exists
from configuration import config
from src.utils.wandb_data_fetcher import download_generations, download_model

if TYPE_CHECKING:
    from src.main.generation import Generation


def wandb_init():
    starting_new_fully_train_run = config.fully_train and not config.resume_fully_train
    starting_new_evolutionary_run = not run_folder_exists(config.run_name) and not config.wandb_run_path
    continuing_a_run = run_folder_exists(config.run_name) or config.resume_fully_train

    if starting_new_fully_train_run or starting_new_evolutionary_run:
        # Either new evolution run or new fully train run
        evo_run_path = config.wandb_run_path
        _new_run()
        if config.fully_train:
            # links the new fully_train wandb run to
            wandb.config['evolution_run_path'] = evo_run_path

    elif continuing_a_run:
        print('resuming')
        _resume_run()
        if config.resume_fully_train:
            print('resuming ft at', wandb.config.evolution_run_path)
            download_generations(run_path=wandb.config.evolution_run_path, replace=True)
            download_model(run_path=wandb.config.evolution_run_path, replace=True)
    else:
        raise Exception("Something went wrong with wandb")

    wandb.config.update(config.__dict__, allow_val_change=True)


def upload_config():
    wandb.config.update(config.__dict__, allow_val_change=True)


def resume_ft_run(reinit=False):
    _resume_run(reinit)
    download_generations(run_path=wandb.config.evolution_run_path, replace=True)
    download_model(run_path=wandb.config.evolution_run_path, replace=True)


def resume_evo_run():
    _resume_run()


def new_ft_run(reinit=False):
    evo_run_path = config.wandb_run_path
    _new_run(reinit)
    if config.fully_train:
        wandb.config['evolution_run_path'] = evo_run_path


def new_evo_run():
    _new_run()


def _resume_run(reinit=False):
    print('resuming wandb run {}'.format(config.wandb_run_path))
    project = 'cdn_fully_train' if config.fully_train else 'cdn'
    wandb_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')
    wandb.init(dir=wandb_dir, project=project, entity='codeepneat', resume=config.wandb_run_path.split('/')[2],
               reinit=reinit, config=config.__dict__)


def _new_run(reinit=False):
    wandb_run_id = config.run_name + str(datetime.date.today()) + '_' + str(randint(1E5, 1E6))
    project = 'cdn_fully_train' if config.fully_train else 'cdn'
    config.wandb_run_path = 'codeepneat/' + project + '/' + wandb_run_id

    dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')

    job_type = 'train'

    if config.fully_train:
        config.wandb_tags += ['FULLY_TRAIN']
    if config.dummy_run:
        config.wandb_tags += ['TEST_RUN']
        job_type = 'test'
    config.wandb_tags = list(set(config.wandb_tags))  # ensures tags are unique

    print(f'starting new wandb run {wandb_run_id}')

    wandb.init(job_type=job_type, project=project, entity='codeepneat', name=config.run_name, tags=config.wandb_tags,
               dir=dir, id=wandb_run_id, reinit=reinit, config=config.__dict__)


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
    raw_bp_sizes = []

    for bp in generation.blueprint_population:
        evaluated_networks = len([fitness[0] for fitness in bp.fitness_raw if fitness[0] != 0])
        n_unevaluated_bps += 1 if evaluated_networks == 0 else 0
        n_large_nets += sum(fitness[0] == 0 for fitness in bp.fitness_raw)
        raw_bp_accs.extend(bp.fitness_raw[0])
        raw_bp_sizes.extend(bp.fitness_raw[1])

    n_unevaluated_mods = 0
    raw_mod_accs = []
    raw_module_sizes = []
    for mod in generation.module_population:
        n_unevaluated_mods += 1 if mod.n_evaluations == 0 else 0
        raw_mod_accs.extend(mod.fitness_raw[0])
        raw_module_sizes.extend(mod.fitness_raw[1])

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

           'blueprint sizes': raw_bp_sizes, 'min blueprint size': min(raw_bp_sizes),
           'avg blueprint size': sum(raw_bp_sizes) / len(raw_bp_sizes),
           'module sizes': raw_module_sizes, 'min module size': min(raw_module_sizes),
           'avg module size': sum(raw_module_sizes) / len(raw_module_sizes),

           'num module species': len(generation.module_population.species),
           'species sizes': [len(spc.members) for spc in generation.module_population.species],
           'unevaluated blueprints': n_unevaluated_bps, 'n_unevaluated_mods': n_unevaluated_mods,
           'large blueprints': n_large_nets, 'speciation threshold': generation.module_population.speciator.threshold,
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
