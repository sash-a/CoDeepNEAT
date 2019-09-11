from torch import nn
from src.CoDeepNEAT.CDNGenomes import BlueprintGenome
from src.NEAT.Genome import Genome
from NEAT.Gene import NodeGene

from src.Phenotype2.Layer import Layer
from src.Phenotype2.AggregationLayer import AggregationLayer

from typing import Type


class Network(nn.Module):
    def __init__(self, blueprint: BlueprintGenome, output_dim=10):
        super().__init__()
        self.blueprint: BlueprintGenome = blueprint

        # TODO: add children

        self.final_layer = nn.Linear(1, output_dim)  # TODO

        # TODO: get params and add optimizer

    def forward(self, input):
        pass

    # TODO should be module nodes topheno
    def create_graph(self, genome: Genome):
        # Discard hanging nodes - i.e nodes that are only connected to either the input or output node
        node_map_from_input = genome.get_reachable_nodes(False)
        node_map_from_output = genome.get_reachable_nodes(True)

        print('from input', node_map_from_input)
        print('from output', node_map_from_output)
        # All non-hanging nodes excluding input and output node
        connected_nodes = node_map_from_input.keys() & node_map_from_output.keys()
        # TODO check that 0 and 1 are IDs of input and output nodes
        connected_nodes.add(0)  # Add input node
        connected_nodes.add(1)  # Add output node
        print('Fully connected nodes:', connected_nodes)

        # Find all nodes with multiple inputs
        multi_input_map = {}  # maps {node id: number of inputs}
        for node_id in connected_nodes:
            num_inputs = sum(list(node_map_from_input.values()), []).count(node_id)
            if num_inputs > 1:
                multi_input_map[node_id] = num_inputs

        # Add nodes with multiple inputs to node_map_from_inputs so that they can be inserted as aggregator nodes
        # Set all multi input nodes (in the values of bp_map) to negative - this represents an aggregator node
        # i.e 3 would have an aggregator node of -3
        for multi_input_node_id in multi_input_map.keys():
            for from_node in node_map_from_input.keys():
                if multi_input_node_id in node_map_from_input[from_node]:
                    idx = node_map_from_input[from_node].index(multi_input_node_id)
                    node_map_from_input[from_node][idx] *= -1

        # Add aggregator nodes as keys and point them to the node they aggregate the inputs for
        # i.e -3 (an agg node) to 3 (a multi input node)
        for multi_input_node_id in multi_input_map.keys():
            node_map_from_input[multi_input_node_id * -1] = [multi_input_node_id]

        print('agg nodes', multi_input_map)
        print('new node map', node_map_from_input)

        input_neat_node = genome.get_input_node()
        input_layer = Layer(input_neat_node)

        def create_layers(current_layer: Layer, current_node_id):
            # Create layers from genome
            if current_node_id == 1:
                return

            for node_id in node_map_from_input[current_node_id]:
                # Check node is a connected node or is an aggregator node before making it a layer
                if node_id not in connected_nodes and node_id >= 0:
                    continue

                if node_id >= 0:
                    new_layer = Layer(genome._nodes[node_id])
                else:
                    new_layer = AggregationLayer(multi_input_map[node_id * -1])

                current_layer.add_module(str(node_id), new_layer)
                create_layers(new_layer, node_id)

        create_layers(input_layer, input_neat_node.id)
        return input_layer


from NEAT.Gene import ConnectionGene
from CoDeepNEAT.CDNGenomes import ModuleGenome
from CoDeepNEAT.CDNNodes import ModulenNEATNode, NodeType

conn0 = ConnectionGene(0, 0, 2)
conn1 = ConnectionGene(1, 0, 3)
conn2 = ConnectionGene(2, 2, 5)
conn3 = ConnectionGene(3, 3, 6)
conn4 = ConnectionGene(4, 6, 4)
conn5 = ConnectionGene(5, 5, 1)
conn6 = ConnectionGene(6, 4, 1)
conn7 = ConnectionGene(7, 3, 5)

# conn3.enabled.set_value(False)

n0 = ModulenNEATNode(0, NodeType.INPUT)
n1 = ModulenNEATNode(1, NodeType.OUTPUT)
n2 = ModulenNEATNode(2)
n3 = ModulenNEATNode(3)
n4 = ModulenNEATNode(4)
n5 = ModulenNEATNode(5)
n6 = ModulenNEATNode(6)

genome = ModuleGenome([conn0, conn1, conn2, conn3, conn4, conn5, conn6, conn7], [n0, n1, n2, n3, n4, n5, n6])

lay = genome.to_phenotype(None)

print()
print(lay)
