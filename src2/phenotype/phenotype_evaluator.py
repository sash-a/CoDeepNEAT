from __future__ import annotations

import threading
from time import time
from typing import TYPE_CHECKING, List

from src2.configuration import config
from src2.phenotype.neural_network.evaluator.evaluator import evaluate
from src2.phenotype.neural_network.neural_network import Network

if TYPE_CHECKING:
    from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome

bp_lock = threading.Lock()


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int], generation_num: int,
                       num_epochs=config.epochs_in_evolution) -> int:
    """
    Parses the blueprint into its phenotype NN
    Handles the assignment of the single/multi obj finesses to the blueprint in parallel
    """
    start = time()
    device = config.get_device()

    constr_start = time()
    model: Network = Network(blueprint, input_size).to(device)
    constr_time = time() - constr_start

    size_start = time()
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_time = time() - size_start

    eval_start = time()
    if model_size > config.max_model_params:
        accuracy = 0
    else:
        accuracy = evaluate(model, num_epochs=num_epochs)
    eval_time = time() - eval_start

    fit_start = time()
    with bp_lock:
        blueprint.update_best_sample_map(model.sample_map, accuracy)
        blueprint.report_fitness([accuracy], module_sample_map=model.sample_map)
        parse_number = blueprint.n_evaluations
    fit_time = time() - fit_start

    other_start = time()
    print("Evaluation of genome:", blueprint.id, "complete with accuracy:", accuracy, "by thread",
          threading.current_thread().name)

    if config.plot_every_genotype:
        blueprint.visualize(parse_number=parse_number,
                            prefix="g" + str(generation_num) + "_" + str(blueprint.id))

    if config.plot_every_phenotype:
        model.visualize(parse_number=parse_number,
                        prefix="g" + str(generation_num) + "_" + str(blueprint.id))
    other_time = time() - other_start
    total_time = time() - start
    print("BP: %i time taken for:\n"
          "Everything: %f\n"
          "Construction: %f\n"
          "Size check: %f\n"
          "Evaluation: %f\n"
          "Fitness reporting: %f\n"
          "Misc: %f"
          % (blueprint.id, total_time, constr_time, size_time, eval_time, fit_time, other_time))

    return model_size
