from __future__ import annotations

from typing import TYPE_CHECKING, List

from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int], generation_num: int,
                       num_epochs=config.epochs_in_evolution) -> int:
    """
    parses the blueprint into its phenotype NN
    handles the assignment of the single/multi obj finesses to the blueprint
    """
    device = config.get_device()
    model: Network = Network(blueprint, input_size).to(device)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_size > config.max_model_params:
        accuracy = 0
    else:
        accuracy = evaluate(model, num_epochs=num_epochs)

    blueprint.update_best_sample_map(model.sample_map, accuracy)
    blueprint.report_fitness([accuracy], module_sample_map=model.sample_map)

    print("Evaluation of genome:", blueprint.id, "complete with accuracy:", accuracy)

    if config.plot_every_genotype:
        blueprint.visualize(parse_number=blueprint.n_evaluations,
                            prefix="g" + str(generation_num) + "_" + str(blueprint.id))

    if config.plot_every_phenotype:
        model.visualize(parse_number=blueprint.n_evaluations,
                        prefix="g" + str(generation_num) + "_" + str(blueprint.id))

    return model_size
