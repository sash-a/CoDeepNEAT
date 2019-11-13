from __future__ import annotations

import copy
import random
from typing import List, Dict, TYPE_CHECKING, Optional, Set, Tuple

from torch import nn

import src2.Genotype.CDN.Nodes.BlueprintNode as BlueprintNode
from src2.Genotype.Mutagen.ContinuousVariable import ContinuousVariable
from src2.Genotype.Mutagen.Mutagen import Mutagen
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node
from src2.Visualisation.GenomeVisualiser import visualise_blueprint_genome
from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Species import Species
    from src2.Phenotype.NeuralNetwork.Layers import Layer
    from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome


class BlueprintGenome(Genome):

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

        self.learning_rate = ContinuousVariable("learning rate", start_range=0.0006, current_value=0.001,
                                                end_range=0.003, mutation_chance=0)
        self.beta1 = ContinuousVariable("beta1", start_range=0.88, current_value=0.9, end_range=0.92, mutation_chance=0)
        self.beta2 = ContinuousVariable("beta2", start_range=0.9988, current_value=0.999, end_range=0.9992,
                                        mutation_chance=0)

        # mapping from species id to the genome id of the module sampled from that species
        self.best_module_sample_map: Optional[Dict[int, int]] = None  # todo empty this at the end of evaluation
        self.best_sample_map_accuracy: float = -1

    def get_modules_used(self):
        """:returns all module ids currently being used by this blueprint. returns duplicates"""
        pass

    def get_all_mutagens(self) -> List[Mutagen]:
        return [self.learning_rate, self.beta1, self.beta2]

    def commit_sample_maps(self):
        """
            commits whatever species->module mapping is in the sample map
            this should be the best sampling found this step
        """
        import src2.main.Singleton as S

        if self.best_module_sample_map is None:
            raise Exception("no sample map attached to blueprint " + repr(self) + " cannot commit")

        for node in self.nodes.values():
            """node may be blueprint or module node"""
            if isinstance(node, BlueprintNode.BlueprintNode):
                """updates the module id value of each node in the genome according to the sample map present"""
                living_speciies = [spc.id for spc in S.instance.module_population.species]
                if node.species_id not in living_speciies:
                    """this species died during the last speciation step"""
                    node.species_id = random.choice(living_speciies)

                if node.species_id in self.best_module_sample_map:
                    """best parent had a node with this spc id"""
                    module_id = self.best_module_sample_map[node.species_id]
                    module = S.instance.module_population[module_id]
                else:
                    """best parent did not have a node with this spc id"""
                    module = None
                    module_id = -1

                node.linked_module_id = module_id if module is not None else -1

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        # print("making blueprint pheno")
        sample_map = {}
        return super().to_phenotype(module_sample_map=sample_map), sample_map

    def visualize(self):
        visualise_blueprint_genome(self, self.best_module_sample_map)

    def end_step(self):
        super().end_step()
        self.commit_sample_maps()
        self.best_module_sample_map = None
        self.best_sample_map_accuracy = -1

    def report_fitness(self, fitnesses, **kwargs):
        # todo thread lock
        super().report_fitness(fitnesses, **kwargs)
        import src2.main.Singleton as S

        sample_map = kwargs["module_sample_map"]

        for node_id in self.get_fully_connected_node_ids():
            node: Node = self.nodes[node_id]
            if not isinstance(node, BlueprintNode.BlueprintNode):
                continue

            if node.species_id not in sample_map:
                raise Exception("sample map"+repr(sample_map)+"doesn't cover all species in blueprint - missing: " + repr(node.species_id))

            module_id = sample_map[node.species_id]
            module = S.instance.module_population[module_id]
            module.report_fitness(fitnesses, **kwargs)

    def update_best_sample_map(self, candidate_map: Dict[int, int], accuracy: int):
        # todo with self.lock:
        if self.best_sample_map_accuracy < accuracy:
            self.best_module_sample_map = candidate_map
            self.best_sample_map_accuracy = accuracy

    def inherit(self, parent: BlueprintGenome):
        self.best_module_sample_map = copy.deepcopy(parent.best_module_sample_map)
