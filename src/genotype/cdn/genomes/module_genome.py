from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.genotype.cdn.nodes.module_node import ModuleNode

from typing import List, Tuple

from runs import runs_manager
from configuration import config
from src.genotype.neat.connection import Connection
from src.genotype.neat.genome import Genome
from src.genotype.neat.node import Node
from src.phenotype.neural_network.layers.layer import Layer
from src.analysis.visualisation.genome_visualiser import get_graph_of


class ModuleGenome(Genome):
    def __int__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        return super().to_phenotype(**kwargs)

    def visualize(self):
        get_graph_of(self).render(directory=runs_manager.get_graphs_folder_path(config.run_name),
                                  view=config.view_graph_plots)

    def get_size_estimate(self):
        node: ModuleNode
        size = 0
        for node_id in self.get_fully_connected_node_ids():
            node = self.nodes[node_id]
            out_features = node.layer_type.get_subvalue('out_features')

            if node.is_conv():
                window_size = node.layer_type.get_subvalue('conv_window_size')

                size += window_size**2 + out_features

            if node.is_linear():
                size += out_features **2
        return size