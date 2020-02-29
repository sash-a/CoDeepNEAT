from typing import Dict, List

from configuration import config
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.neural_network import Network


def get_model_of_target_size(blueprint: BlueprintGenome, sample_map: Dict[int, int], original_model_size,
                             input_size: List[int], target_size=-1) -> Network:
    if target_size == -1:
        target_size = config.target_network_size

    feature_mulitplication_first_guess = pow(target_size / original_model_size, 0.5)
    feature_mult_best_approximation = refine_feature_multiplication_guess(feature_mulitplication_first_guess, blueprint,
                                                                          sample_map, input_size, target_size)

    model: Network = Network(blueprint, input_size, feature_multiplier=feature_mult_best_approximation,
                             sample_map=sample_map, allow_module_map_ignores=False).to(config.get_device())
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("targeting size using feature mult:", feature_mult_best_approximation, "original size:", original_model_size,
          "normalised size:", model_size, "target:", target_size, "change ratio:",
          (model_size / original_model_size), "target ratio:", (model_size / target_size))
    return model


def refine_feature_multiplication_guess(feature_mulitplication_guess, blueprint: BlueprintGenome, sample_map: Dict[int, int],
                                        input_size: List[int], target_size ,remaining_tries=5, best_guess=-1, best_target_ratio=-1):
    """method to iteratively refine the FM guess to get closest to target size"""
    if remaining_tries == 0:
        return best_guess
    model: Network = Network(blueprint, input_size, feature_multiplier=feature_mulitplication_guess,
                             sample_map=sample_map, allow_module_map_ignores=False).to(config.get_device())
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_ratio = model_size/target_size
    # ideal target ratio is 1, >1 means decrease FM, <1 means increase FM
    if best_guess == -1 or abs(1-target_ratio) < abs(1-best_target_ratio):
        # no best or new guess is better than best
        best_guess = feature_mulitplication_guess
        best_target_ratio = target_ratio

    # print("fm guess:",feature_mulitplication_guess,"target ratio:",target_ratio, "best guess:",best_guess,"best rat:",best_target_ratio)
    adjustment_factor = 0.2 + (remaining_tries/20) # how big of a jump to make - decrease each guess
    next_guess = best_guess / pow(best_target_ratio, adjustment_factor)
    return refine_feature_multiplication_guess(next_guess, blueprint, sample_map, input_size, target_size,
                                               remaining_tries=remaining_tries-1,
                                               best_guess=best_guess, best_target_ratio=best_target_ratio)
