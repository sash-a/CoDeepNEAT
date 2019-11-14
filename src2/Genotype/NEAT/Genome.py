from __future__ import annotations

import copy
from typing import Dict, List, AbstractSet, Union, TYPE_CHECKING, Set, Tuple

import sys
import threading
from torch import nn

from src2.Genotype.NEAT.GraphGenome import GraphGenome
from src2.Genotype.NEAT.MutationRecord import MutationRecords
from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer
from src2.Phenotype.NeuralNetwork.Layers.Layer import Layer
from src2.Configuration import config
from src2.Genotype.Mutagen.Mutagen import Mutagen

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Connection import Connection
    from src2.Genotype.NEAT.Node import Node
    from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
    from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode


class Genome(GraphGenome):
    _id_counter = 0

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)
        self.id = Genome._id_counter
        Genome._id_counter += 1

        self.rank = 0  # The order of this genome when ranked by fitness values
        self.uses = 0  # The numbers of times this genome is used
        self.fitness_values: List[int] = [-(sys.maxsize - 1)]

        self.lock = threading.RLock()

    def __deepcopy__(self, memodict={}):
        cp = type(self)(copy.deepcopy(list(self.nodes.values())), copy.deepcopy(list(self.connections.values())))

        cp.id = self.id
        cp.rank = self.rank
        cp.uses = self.uses
        cp.fitness_values = self.fitness_values

        return cp

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __gt__(self, other):
        return self.rank > other.rank

    def __lt__(self, other):
        return self.rank < other.rank

    def report_fitness(self, fitnesses, **kwargs):
        """updates the fitnesses stored with a new given fitness"""
        with self.lock:
            if self.fitness_values is None or not self.fitness_values:
                self.fitness_values = [0 for _ in fitnesses]

            if config.fitness_aggregation == 'avg':
                for i, fitness in enumerate(fitnesses):
                    if self.fitness_values[i] is None:
                        self.fitness_values[i] = 0
                    self.fitness_values[i] = (self.fitness_values[i] * self.uses + fitness) / (self.uses + 1)
                self.uses += 1
            elif config.fitness_aggregation == 'max':
                for i, fitness in enumerate(fitnesses):
                    if self.fitness_values[i] is None:
                        self.fitness_values[i] = 0
                    self.fitness_values[i] = max(self.fitness_values[i], fitness)
            else:
                raise Exception("Unexpected fitness aggregation type: " + repr(config.fitness_aggregation))

    def end_step(self):
        """Resets all necessary values for next the generation"""
        self.uses = 0
        if self.fitness_values is not None:
            self.fitness_values = [0 for _ in self.fitness_values]

    def distance_to(self, other) -> float:
        return self.get_topological_distance(other)

    def validate(self) -> bool:
        connected = self._validate_traversal(self.get_input_node().id, self.get_traversal_dictionary(True), set())
        return connected and not self.has_cycle()

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        # print("making genome pheno")
        multi_input_map: Dict[int, int] = self.get_multi_input_nodes()
        node_traversal: Dict[int, List[int]] = self.get_reachable_nodes(False)
        connected_nodes: Set[int] = self.get_fully_connected_node_ids()
        # Add nodes with multiple inputs to node_map_from_inputs so that they can be inserted as aggregator nodes
        # Aggregator nodes are represented as negative the node they are aggregating for
        # Set all multi input nodes (node_map.values) to negative
        for multi_input_node_id in multi_input_map.keys():
            """for each node with multiple inputs"""
            for from_node in node_traversal.keys():
                """for each from node except the output node"""
                if multi_input_node_id in node_traversal[from_node]:
                    """if from node leads into m_i_n"""
                    idx = node_traversal[from_node].index(multi_input_node_id)
                    node_traversal[from_node][idx] *= -1  # Mark a node as agg node by making it negative

        # Add aggregator nodes as keys and point them to the node they aggregate the inputs for
        # i.e -3 (an agg node) to 3 (used to be multi input node, now single input)
        for multi_input_node_id in multi_input_map.keys():
            node_traversal[multi_input_node_id * -1] = [multi_input_node_id]

        agg_layers: Dict[int, AggregationLayer] = {}  # map from node marked as agg to the agg layer pheno

        blueprint_node_id = kwargs["blueprint_node_id"] if "blueprint_node_id" in kwargs else None

        # input node of the input module and output node of the input module
        first_layer, starting_layer = self.get_input_node().convert_node(**kwargs, node_id=blueprint_node_id)
        output_layers = self.get_output_node().convert_node(**kwargs, node_id=blueprint_node_id)

        def create_and_link_layers(parent_nn_output: Layer, parent_node_id) -> None:
            """
            :param parent_nn_output: the output layer of the module picked by the parent blueprint node
            """
            if parent_node_id not in node_traversal:
                if parent_node_id != 1:
                    raise Exception("node with id " + str(parent_node_id) + " is a dead end")
                """is blueprint output node"""
                return

            for child_node_id in node_traversal[parent_node_id]:
                # Check node is a connected node or is an aggregator node before making it a layer
                if child_node_id not in connected_nodes and child_node_id >= 0:
                    continue

                input_layer: nn.Module
                output_layer: nn.Module

                if child_node_id >= 0:  # not an aggregator node
                    # Creates a new module
                    node: Node = self.nodes[child_node_id]
                    input_layer, output_layer = node.convert_node(**kwargs, node_id=blueprint_node_id)
                    # Use already created output module if child is output node
                    if node.is_output_node():
                        input_layer, output_layer = output_layers

                    create_and_link_layers(output_layer, child_node_id)
                elif child_node_id in agg_layers:
                    input_layer = agg_layers[child_node_id]  # only create an aggregation layer once
                else:
                    # Create aggregation layer if not already created and node_id is negative
                    if blueprint_node_id is not None:
                        """this is a module agg node"""
                        name = str(blueprint_node_id) + "_agg(" + str(-1 * child_node_id) + ")"
                    else:
                        """blueprint level agg node"""
                        name = "agg(" + str(-1 * child_node_id) + ")"

                    input_layer = AggregationLayer(multi_input_map[child_node_id * -1], name)
                    agg_layers[child_node_id] = input_layer
                    create_and_link_layers(input_layer, child_node_id)

                # Connect output of parent to input of child
                parent_nn_output.add_child(str(child_node_id), input_layer)

        # starts the recursive call for creating modules
        create_and_link_layers(starting_layer, self.get_input_node().id)
        return first_layer, output_layers[1]  # return first and last layer in the full NN

    def visualize(self):
        pass

    def get_all_mutagens(self) -> List[Mutagen]:
        return []

    def inherit(self, parent):
        pass
