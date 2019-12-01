from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Tuple

from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome

parse_number_map: Dict[Tuple[int, int], int] = {}  # maps from (gen,genoID) to parseNum


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int]):
    """
    parses the blueprint into its phenotype NN
    handles the assignment of the single/multi obj finesses to the blueprint
    """
    parse_number, generation_number = get_parse_gen_num(blueprint)

    model: Network = Network(blueprint, input_size)
    device = config.get_device()
    model.to(device)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable model params', model_size)

    if model_size > config.max_model_params:
        accuracy = 0
    else:
        accuracy = evaluate(model)

    blueprint.update_best_sample_map(model.sample_map, accuracy)
    blueprint.report_fitness([accuracy], module_sample_map=model.sample_map)

    print("Evaluation of genome:", blueprint.id, "complete with accuracy:", accuracy)

    if config.plot_every_genotype:
        blueprint.visualize(parse_number= parse_number, prefix= "g" + str(generation_number) + "_")
    if config.plot_every_phenotype:
        model.visualize(parse_number=parse_number, prefix= "g" + str(generation_number) + "_")

    return blueprint


def get_parse_gen_num(blueprint: BlueprintGenome):
    import src2.main.Singleton as Singleton

    key = (Singleton.instance.generation_number, blueprint.id)
    if key not in parse_number_map.keys():
        parse_number_map[key] = 0
        return 0, Singleton.instance.generation_number

    parse_number_map[key] += 1
    return parse_number_map[key], Singleton.instance.generation_number
