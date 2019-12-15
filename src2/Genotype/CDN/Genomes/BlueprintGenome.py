from __future__ import annotations

import copy
import random
import threading
from typing import List, Dict, TYPE_CHECKING, Optional, Tuple

import src2.Genotype.CDN.Nodes.BlueprintNode as BlueprintNode
from src2.Configuration import config
from src2.Genotype.Mutagen.ContinuousVariable import ContinuousVariable
from src2.Genotype.Mutagen.Mutagen import Mutagen
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node
from src2.Visualisation.GenomeVisualiser import visualise_blueprint_genome

# TESTING
from src.CoDeepNEAT.CDNGenomes.BlueprintGenome import BlueprintGenome as BPG_old

if TYPE_CHECKING:
    from src2.Phenotype.NeuralNetwork.Layers import Layer


class BlueprintGenome(Genome):

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

        # TODO mutate, CDN has ranges in 2017 paper
        self.learning_rate = ContinuousVariable("learning rate", start_range=0.0006, current_value=0.001,
                                                end_range=0.003, mutation_chance=0)
        self.beta1 = ContinuousVariable("beta1", start_range=0.88, current_value=0.9, end_range=0.92, mutation_chance=0)
        self.beta2 = ContinuousVariable("beta2", start_range=0.9988, current_value=0.999, end_range=0.9992,
                                        mutation_chance=0)

        # mapping from species id to the genome id of the module sampled from that species
        self.best_module_sample_map: Optional[Dict[int, int]] = None  # todo empty this at the end of evaluation
        self.best_sample_map_accuracy: float = -1

    def __deepcopy__(self, memodict={}):
        cp = super().__deepcopy__()

        cp.learning_rate = copy.deepcopy(self.learning_rate)
        cp.beta1 = copy.deepcopy(self.beta1)
        cp.beta2 = copy.deepcopy(self.beta2)
        cp.best_module_sample_map = copy.deepcopy(self.best_module_sample_map)
        cp.best_sample_map_accuracy = copy.deepcopy(self.best_sample_map_accuracy)

        return cp

    # def __repr__(self):
    #     return '\n'.join([str(self.learning_rate.value), str(self.beta1.value), str(self.beta2.value),
    #                       str(self.best_module_sample_map), str(self.nodes.keys()), str(self.nodes.values()),
    #                       str(id(self))])

    def __getstate__(self):
        d = dict(self.__dict__)
        if 'lock' in d:
            del d['lock']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self.__dict__['lock'] = threading.RLock()

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
            return

        for node in self.nodes.values():
            """node may be blueprint or module node"""
            if isinstance(node, BlueprintNode.BlueprintNode):
                """updates the module id value of each node in the genome according to the sample map present"""
                living_species = [spc.id for spc in S.instance.module_population.species]
                if node.species_id not in living_species:
                    """this species died during the last speciation step"""
                    node.species_id = random.choice(living_species)

                if node.species_id in self.best_module_sample_map:
                    """
                        best parent had a node with this spc id 
                        try to find the mapped module if it survived
                    """
                    module_id = self.best_module_sample_map[node.species_id]
                    module = S.instance.module_population[module_id]
                else:
                    """best parent did not have a node with this spc id"""
                    module = None
                    module_id = -1

                if config.use_module_retention:
                    node.linked_module_id = module_id if module is not None else -1

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        sample_map = {}
        if "prescribed_sample_map" in kwargs and kwargs["prescribed_sample_map"] is not None:
            sample_map = kwargs["prescribed_sample_map"]

        return super().to_phenotype(module_sample_map=sample_map, **kwargs), sample_map

    def visualize(self, parse_number=-1, prefix=""):
        visualise_blueprint_genome(self, self.best_module_sample_map, parse_number=parse_number, prefix=prefix)

    def before_step(self):
        super().before_step()
        self.best_module_sample_map = None

    def end_step(self):
        super().end_step()
        self.commit_sample_maps()
        self.best_sample_map_accuracy = -1

    def report_fitness(self, fitnesses: List[float], **kwargs):
        super().report_fitness(fitnesses)
        import src2.main.Singleton as Singleton

        with self.lock:
            sample_map = kwargs["module_sample_map"]

            for node_id in self.get_fully_connected_node_ids():
                node: Node = self.nodes[node_id]
                if not isinstance(node, BlueprintNode.BlueprintNode):
                    """module node"""
                    continue

                if node.species_id not in sample_map:
                    raise Exception(
                        "sample map" + repr(sample_map) + "doesn't cover all species in blueprint - missing: " + repr(
                            node.species_id))

                module_id = sample_map[node.species_id]
                module = Singleton.instance.module_population[module_id]
                module.report_fitness(fitnesses, **kwargs)

    def update_best_sample_map(self, candidate_map: Dict[int, int], accuracy: int):
        with self.lock:
            if self.best_sample_map_accuracy < accuracy:
                self.best_module_sample_map = candidate_map
                self.best_sample_map_accuracy = accuracy

    def inherit(self, parent: BlueprintGenome):
        self.best_module_sample_map = copy.deepcopy(parent.best_module_sample_map)

    def old(self) -> BPG_old:
        old_nodes = []
        old_conns = []

        for node in self.nodes.values():
            old_nodes.append(node.old())

        for connection in self.connections.values():
            old_conns.append(connection.old())

        return BPG_old(old_conns, old_nodes)

