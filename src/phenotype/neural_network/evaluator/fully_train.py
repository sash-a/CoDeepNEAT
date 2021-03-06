from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import src.main.singleton as S
from configuration import config, internal_config
from runs.runs_manager import get_fully_train_folder_path
from src.analysis.reporters.logger_reporter import LoggerReporter
from src.analysis.reporters.reporter_set import ReporterSet
from src.analysis.reporters.wandb_ft_reporter import WandbFTReporter
from src.analysis.run import get_run
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src.phenotype.neural_network.evaluator.evaluator import evaluate, RETRY
from src.phenotype.neural_network.feature_multiplication import get_model_of_target_size
from src.phenotype.neural_network.neural_network import Network
from src.utils.mp_utils import create_eval_pool

if TYPE_CHECKING:
    from src.analysis.run import Run

MAX_RETRIES = 3


def fully_train(run_name):
    """
    Loads and trains from a saved run.
    This will parallelize all training of all the best networks across the given config.ft_feature_multipliers. i.e each
    different feature multiplier will gets its own process and own gpu if available.
    :param run_name: name of the evolutionary run
    """
    print('Fully training...')
    internal_config.ft_started = True
    internal_config.state = 'ft'

    run: Run = get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(config.fully_train_best_n_blueprints)
    in_size = get_data_shape()

    print(f'best blueprints ({len(best_blueprints)}): {[b[0].id for b in best_blueprints]}')

    with create_eval_pool(None) as pool:
        futures = []
        for feature_mul in config.ft_feature_multipliers:
            for i, (blueprint, gen_num) in enumerate(best_blueprints, 1):
                futures += [
                    pool.submit(eval_with_retries, run, blueprint, gen_num, in_size, feature_mul, i)
                ]

        for future in futures:  # consuming the futures
            print(future.result())

    # Finished if number of saved models is the expected max number of models
    internal_config.finished = len(os.listdir(get_fully_train_folder_path(config.run_name))) == \
                               len(config.ft_feature_multipliers) * config.fully_train_best_n_blueprints
    internal_config.state = 'finished' if internal_config.finished else 'ft'


def eval_with_retries(run: Run, blueprint: BlueprintGenome, gen_num: int, in_size: List[int], feature_mul: int,
                      best: int):
    """
    Evaluates a run and automatically retries is accuracy is not keeping up with the accuracy achieved in evolution
    """
    print(f'Starting ft for fm{feature_mul} best{best}')
    save_suffix = f'-best-{best}'
    accuracy = RETRY
    remaining_retries = MAX_RETRIES
    model: Optional[Network] = None
    while accuracy == RETRY and remaining_retries >= 0:
        model = _create_model(run, blueprint, gen_num, in_size, feature_mul)

        if os.path.exists(model.save_location(save_suffix)):
            print(f'Found file {model.save_location(save_suffix)}. Therefore not ft: fm{feature_mul} best{best}')
            return  # If the model was saved that means it has completed its fully training schedule

        attempt_number = MAX_RETRIES - remaining_retries
        reporter = ReporterSet(LoggerReporter(feature_mul, best, attempt_number), WandbFTReporter(feature_mul, best))
        reporter.on_start_train(blueprint)

        print(f'ft attempt {attempt_number} for fm{feature_mul} best{best}')

        # If all trains are bad then will not get a FT accuracy for this blueprint
        if remaining_retries > 0:
            accuracy = evaluate(model,
                                config.fully_train_max_epochs,
                                training_target=blueprint.max_acc,
                                attempt=attempt_number,
                                reporter=reporter)

        reporter.on_end_train(blueprint, accuracy)

        if accuracy == RETRY:
            print(f'Retrying fully training for fm{feature_mul} best{best}')
            internal_config.ft_epoch = 0

        remaining_retries -= 1

    if model is not None:
        model.save(save_suffix)  # even if it fails all retries then save it

    if accuracy == RETRY:
        print(f'Accuracy could not keep up with evolution after {MAX_RETRIES}, therefore discarding it '
              f'(fm{feature_mul} best{best})')
    else:
        print(f'Achieved a final accuracy of: {accuracy * 100} (fm{feature_mul} best{best})')


def _create_model(run: Run, blueprint: BlueprintGenome, gen_num, in_size, target_feature_multiplier) -> Network:
    S.instance = run.generations[gen_num]
    modules = run.get_modules_for_blueprint(blueprint)
    model: Network = Network(blueprint, in_size, sample_map=blueprint.best_module_sample_map,
                             allow_module_map_ignores=False, feature_multiplier=1,
                             target_feature_multiplier=target_feature_multiplier).to(config.get_device())

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sample_map = model.sample_map

    if target_feature_multiplier != 1:
        model = get_model_of_target_size(blueprint, sample_map, model_size, in_size,
                                         target_size=model_size * target_feature_multiplier)
        model.target_feature_multiplier = target_feature_multiplier

    print("Blueprint: {}\nModules: {}\nSample map: {}\n Species used: {}"
          .format(blueprint,
                  modules,
                  blueprint.best_module_sample_map,
                  list(set(
                      [blueprint.nodes[node_id].species_id for node_id in blueprint.get_fully_connected_node_ids()
                       if type(blueprint.nodes[node_id]) == BlueprintNode]))))
    print("Training model which scored: {} in evolution , with {} parameters with feature mult: {}\n"
          .format(blueprint.max_acc, model.size(), target_feature_multiplier))

    return model


# Unused
def _load_model(dummy_bp: BlueprintGenome, run: Run, gen_num: int, in_size) -> Network:
    if not config.resume_fully_train:
        raise Exception('Calling resume training, but config.resume_fully_train is false')

    S.instance = run.generations[gen_num]
    model: Network = Network(dummy_bp, in_size, sample_map=dummy_bp.best_module_sample_map,
                             allow_module_map_ignores=False).to(config.get_device())
    model.load()

    return model
