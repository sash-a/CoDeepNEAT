from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

import math

from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network
import src.Validation.Validation as Val

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int], generation_num: int,
                       num_epochs=config.epochs_in_evolution) -> int:
    """
    parses the blueprint into its phenotype NN
    handles the assignment of the single/multi obj finesses to the blueprint
    """
    ignore_species = -1
    if blueprint.n_evaluations > 0 and config.module_map_ignore_chance > random.random() and config.use_module_retention:
        ignore_species = forget_modules(blueprint)

    device = config.get_device()
    model: Network = Network(blueprint, input_size, ignore_species = ignore_species).to(device)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_size > config.max_model_params:
        accuracy = 0
    else:
        accuracy = evaluate(model, num_epochs=num_epochs)

    blueprint.update_best_sample_map(model.sample_map, accuracy)
    blueprint.report_fitness([accuracy], module_sample_map=model.sample_map)

    old = blueprint.old()

    # Val.get_accuracy_estimate_for_network()

    print("Evaluation of genome:", blueprint.id, "complete with accuracy:", accuracy)

    if config.plot_every_genotype:
        blueprint.visualize(parse_number=blueprint.n_evaluations,
                            prefix="g" + str(generation_num) + "_" + str(blueprint.id))

    if config.plot_every_phenotype:
        model.visualize(parse_number=blueprint.n_evaluations,
                        prefix="g" + str(generation_num) + "_" + str(blueprint.id))

    return model_size


def forget_modules(blueprint: BlueprintGenome):
    """
        forget module maps with a probability based
        on how fully mapped the blueprint is
    """

    nodes = blueprint.nodes.values()
    species_ids = set([node.species_id for node in nodes])
    mapped_species = set([node.species_id for node in nodes if node.linked_module_id != -1])

    map_frac = len(mapped_species)/len(species_ids)
    if (random.random() < math.pow(map_frac, 1.5)) or map_frac == 1:
        """fully mapped blueprints are guaranteed to lose a mapping"""
        ignore_species_id = random.choice(list(species_ids))
        # print("ignoring species map for", ignore_species_id)
        return ignore_species_id
    return -1