from __future__ import annotations

from typing import List, Dict, TYPE_CHECKING, Optional

from torch import nn

from Genotype.CDN.Nodes import BlueprintNode
from Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.Mutagen.ContinuousVariable import ContinuousVariable
from src2.Genotype.Mutagen.Mutagen import Mutagen
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node
from Phenotype.NeuralNetwork.Layers import AggregationLayer

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Species import Species
    from Phenotype.NeuralNetwork.Layers import Layer
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
        self.module_sample_map: Optional[Dict[int, int]] = None

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
        for node in self.nodes.values():
            """node may be blueprint or module node"""
            if isinstance(node, BlueprintNode):
                """updates the module id value of each node in the genome according to the sample map present"""
                node.linked_module_id = self.module_sample_map[node.species_id]

    def to_phenotype(self, module_species: List[Species]):
        multi_input_map = self.get_multi_input_nodes()
        node_map = self.get_reachable_nodes(False)
        connected_nodes = self.get_fully_connected_node_ids()
        # Add nodes with multiple inputs to node_map_from_inputs so that they can be inserted as aggregator nodes
        # Aggregator nodes are represented as negative the node they are aggregating for
        # Set all multi input nodes (node_map.values) to negative
        for multi_input_node_id in multi_input_map.keys():
            for from_node in node_map.keys():
                if multi_input_node_id in node_map[from_node]:
                    idx = node_map[from_node].index(multi_input_node_id)
                    node_map[from_node][idx] *= -1  # Mark a node as agg node by making it negative

        # Add aggregator nodes as keys and point them to the node they aggregate the inputs for
        # i.e -3 (an agg node) to 3 (used to be multi input node, now single input)
        for multi_input_node_id in multi_input_map.keys():
            node_map[multi_input_node_id * -1] = [multi_input_node_id]

        agg_layers = {}
        # input node of the input module and output node of the input module
        # TODO pick module
        input_module_input, input_module_output = self.get_input_node().pick_module(self.species_module_index_map,
                                                                                    module_species).to_phenotype(None,
                                                                                                                 0)
        output_module = self.get_output_node().pick_module(self.species_module_index_map, module_species).to_phenotype(
            None, 1)

        def create_modules(parent_nn_output: Layer, parent_bp_node_id):
            if parent_bp_node_id not in node_map:
                return

            for child_node_id in node_map[parent_bp_node_id]:
                # Check node is a connected node or is an aggregator node before making it a layer
                if child_node_id not in connected_nodes and child_node_id >= 0:
                    continue

                nn_input_layer: nn.Module
                nn_output_layer: nn.Module

                if child_node_id >= 0:
                    # Creates a new module
                    bp_node: BlueprintNode = self.nodes[child_node_id]
                    # TODO pick module
                    neat_module: ModuleGenome = bp_node.pick_module(self.species_module_index_map, module_species)
                    # Use already created output module if child is output node
                    nn_input_layer, nn_output_layer = neat_module.to_phenotype(bp_node.id) \
                        if not bp_node.is_output_node() else output_module
                    create_modules(nn_output_layer, child_node_id)
                elif child_node_id in agg_layers:
                    nn_input_layer = agg_layers[child_node_id]  # only create an aggregation layer once
                else:
                    # Create aggregation layer if not already created and node_id is negative
                    nn_input_layer = AggregationLayer(multi_input_map[child_node_id * -1],
                                                      str(child_node_id) + '_' + str(child_node_id))
                    agg_layers[child_node_id] = nn_input_layer
                    create_modules(nn_input_layer, child_node_id)

                # Connect output of parent to input of child
                parent_nn_output.add_child(str(child_node_id), nn_input_layer)

        create_modules(input_module_output, self.get_input_node().id)  # starts the recursive call for creating modules
        return input_module_input, output_module[1]
