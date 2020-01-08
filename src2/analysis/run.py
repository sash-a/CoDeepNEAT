from __future__ import annotations

import heapq
from typing import List, Tuple, Dict, Set
from typing import TYPE_CHECKING

from runs import runs_manager

if TYPE_CHECKING:
    from src2.main.generation import Generation
    from src2.configuration import configuration
    from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
    from src2.genotype.cdn.genomes.module_genome import ModuleGenome


class Run:
    def __init__(self, generations: List[Generation], configuration: configuration):
        self.generations: List[Generation] = generations
        self.config: configuration = configuration

    def get_genome_at(self, generation_number: int, genome_id: int):
        if generation_number > len(self.generations):
            raise Exception("gen index out of bounds")

        return self.generations[generation_number][genome_id]

    def get_first_instance_of_genome(self, genome_id: int):
        for gen in self.generations:
            mem = gen[genome_id]
            if mem is not None:
                return mem

        return None

    def get_most_accurate_blueprints(self, n=1) -> List[Tuple[BlueprintGenome, int]]:
        """:return: a list of tuples (genome,gen_no) of the best bp genomes and their best gen"""
        blueprints = []
        for gen in self.generations:
            if gen is not None:
                bps = list(gen.blueprint_population)
                blueprints += list(zip(bps, [gen.generation_number] * len(bps)))
            else:
                print("Null generation found")

        most_accurate = heapq.nlargest(n, blueprints, key=lambda x: x[0].max_accuracy)
        return most_accurate

    def get_modules_for_blueprint(self, blueprint: BlueprintGenome) -> Dict[int, ModuleGenome]:
        modules: Dict[int, ModuleGenome] = {}
        module_ids: Set[int] = set()
        for node in blueprint.nodes.values():
            if node.linked_module_id == -1:
                if node.species_id not in blueprint.best_module_sample_map.keys():
                    raise Exception("unlinked blueprint node")
                else:
                    module_ids.add(blueprint.best_module_sample_map[node.species_id])
            else:
                module_ids.add(node.linked_module_id)
        for module_id in module_ids:
            module = self.get_first_instance_of_genome(module_id)  # modules don't change over gens
            if module is None:
                raise Exception("cannot find module in run")
            modules[module_id] = module

        return modules


def get_run(run_name, config_name="config"):
    last_gen = runs_manager.get_latest_generation(run_name)
    generations: List[Generation] = []
    for i in range(last_gen + 1):
        generation: Generation = runs_manager.load_generation(i, run_name)
        generations.append(generation)

    configuration: configuration = runs_manager.load_config(run_name=run_name, config_name=config_name)

    return Run(generations, configuration)


if __name__ == "__main__":
    run = get_run("base")
    print([x[0].max_accuracy for x in run.get_most_accurate_blueprints(5)])
