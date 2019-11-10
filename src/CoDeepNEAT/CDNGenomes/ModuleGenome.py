import copy
import math

from src.CoDeepNEAT.CDNNodes.ModuleNode import ModuleNEATNode
from src.Config import NeatProperties as Props
from src.NEAT.Genome import Genome
from src.Phenotype.ModuleNode import ModuleNode
from src2.Phenotype.NeuralNetwork.Layers import Layer
from src2.Phenotype.NeuralNetwork.Layers import AggregationLayer


class ModuleGenome(Genome):
    """the module variation of the genome class."""

    def __init__(self, connections, nodes):
        super(ModuleGenome, self).__init__(connections, nodes)
        self.module_node = None  # the module node created from this gene

    def __eq__(self, other):
        if not isinstance(other, ModuleGenome):
            raise TypeError(str(type(other)) + ' cannot be equal to ModuleGenome')

        self_ids = [nodeid for nodeid in self._nodes.keys()] + [connid for connid in self._connections.keys()]
        other_ids = [nodeid for nodeid in other._nodes.keys()] + [connid for connid in other._connections.keys()]

        self_attribs = []
        for node in self._nodes.values():
            for mutagen in node.get_all_mutagens():
                self_attribs.extend(mutagen.get_all_sub_values())

        other_attribs = []
        for node in other._nodes.values():
            for mutagen in node.get_all_mutagens():
                other_attribs.extend(mutagen.get_all_sub_values())

        return self_ids == other_ids and self_attribs == other_attribs

    def __hash__(self):
        self_ids = [nodeid for nodeid in self._nodes.keys()] + [connid for connid in self._connections.keys()]
        attribs = []
        for node in self._nodes.values():
            for mutagen in node.get_all_mutagens():
                attribs.extend(mutagen.get_all_sub_values())

        return hash(tuple(self_ids + attribs))

    def to_module(self):
        """
        returns the stored module_node of this gene, or generates and returns it if module_node is null
        :return: the module graph this individual represents
        """
        if self.module_node is not None:
            return copy.deepcopy(self.module_node)

        module = super().to_phenotype(ModuleNode)
        self.module_node = module
        return copy.deepcopy(module)

    def to_phenotype(self, Phenotype, bp_id):
        multi_input_map = self.get_multi_input_nodes()
        node_map = self.get_reachable_nodes(False)
        connected_nodes = self.get_fully_connected_nodes()
        # Add nodes with multiple inputs to node_map_from_inputs so that they can be inserted as aggregator nodes
        # Aggregator nodes are represented as negative the node they are aggregating for
        # Set all multi input nodes (node_map.values) to negative
        for multi_input_node_id in multi_input_map.keys():
            for from_node in node_map.keys():
                if multi_input_node_id in node_map[from_node]:
                    idx = node_map[from_node].index(multi_input_node_id)
                    node_map[from_node][idx] *= -1

        # Add aggregator nodes as keys and point them to the node they aggregate the inputs for
        # i.e -3 (an agg node) to 3 (used to be multi input node, now single input)
        for multi_input_node_id in multi_input_map.keys():
            node_map[multi_input_node_id * -1] = [multi_input_node_id]

        agg_layers = {}  # maps {node_id : AggregatorLayer}

        input_neat_node = self.get_input_node()
        input_layer = Layer(input_neat_node, str(bp_id) + '_0')
        output_layer = Layer(self.get_output_node(), str(bp_id) + '_1')

        def create_layers(parent_layer: Layer, parent_node_id: int):
            if parent_node_id not in node_map:
                return

            for child_node_id in node_map[parent_node_id]:
                # Check node is a connected node or is an aggregator node before making it a layer
                if child_node_id not in connected_nodes and child_node_id >= 0:
                    continue

                if child_node_id >= 0:
                    # Creates a new layer
                    neat_node: ModuleNEATNode = self._nodes[child_node_id]
                    # Use already created output layer if child is output node
                    new_layer = Layer(neat_node, str(bp_id) + '_' + str(
                        child_node_id)) if not neat_node.is_output_node() else output_layer
                    create_layers(new_layer, child_node_id)
                elif child_node_id in agg_layers:
                    new_layer = agg_layers[child_node_id]  # only create an aggregation layer once
                else:
                    # Create aggregation layer if not already created and node_id is negative
                    new_layer = AggregationLayer(multi_input_map[child_node_id * -1],
                                                 str(bp_id) + '_' + str(child_node_id))
                    agg_layers[child_node_id] = new_layer
                    create_layers(new_layer, child_node_id)

                # Add new layer as a child of current layer
                parent_layer.add_child(str(bp_id) + '_' + str(child_node_id), new_layer)

        create_layers(input_layer, input_neat_node.id)  # starts the recursive call for creating layers

        return input_layer, output_layer

    def distance_to(self, other):
        """the similarity metric used by modules for speciation"""
        if type(self) != type(other):
            raise TypeError('Trying finding distance from Module genome to ' + str(type(other)))

        attrib_dist = self.get_attribute_distance(other)
        topology_dist = self.get_topological_distance(other)

        return math.sqrt(attrib_dist * attrib_dist + topology_dist * topology_dist)

    def get_attribute_distance(self, other):
        """done mutagen wise - combines the distance of each corresponding mutagen based on their values"""
        if not isinstance(other, ModuleGenome):
            raise TypeError('Expected type of ModuleGenome, received type: ' + str(type(other)))

        attrib_dist = 0
        common_nodes = self._nodes.keys() & other._nodes.keys()

        for node_id in common_nodes:
            self_node, other_node = self._nodes[node_id], other._nodes[node_id]
            for self_mutagen, other_mutagen in zip(self_node.get_all_mutagens(), other_node.get_all_mutagens()):
                attrib_dist += self_mutagen.distance_to(other_mutagen)

        attrib_dist /= len(common_nodes)
        return attrib_dist

    def mutate(self, mutation_record, attribute_magnitude=1, topological_magnitude=1, module_population=None, gen=-1):
        return super()._mutate(mutation_record, Props.MODULE_NODE_MUTATION_CHANCE, Props.MODULE_CONN_MUTATION_CHANCE,
                               attribute_magnitude=attribute_magnitude, topological_magnitude=topological_magnitude)

    def inherit(self, genome):
        pass

    def end_step(self, generation=None):
        super().end_step()
        self.module_node = None

    # def __repr__(self):
    #     return str(hash(self))

    def get_comlexity(self):
        """approximates the parameter size of this module"""
        complexity = 0
        for node in self._nodes.values():
            complexity += node.get_complexity()
        return complexity
