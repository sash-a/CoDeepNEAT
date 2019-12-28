from typing import List, Tuple

from runs import runs_manager
from src.CoDeepNEAT.CDNGenomes.ModuleGenome import ModuleGenome as ModuleGenome_Old
from src2.configuration import config
from src2.genotype.neat.connection import Connection
from src2.genotype.neat.genome import Genome
from src2.genotype.neat.node import Node
from src2.phenotype.neural_network.layers.layer import Layer
from src2.analysis.visualisation.genome_visualiser import get_graph_of


class ModuleGenome(Genome):
    def __int__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        return super().to_phenotype(**kwargs)

    def visualize(self):
        get_graph_of(self).render(directory=runs_manager.get_graphs_folder_path(config.run_name),
                                  view=config.view_graph_plots)

    def old(self) -> ModuleGenome_Old:
        old_nodes = []
        old_conns = []

        for node in self.nodes.values():
            old_nodes.append(node.old())

        for connection in self.connections.values():
            old_conns.append(connection.old())

        return ModuleGenome_Old(old_conns, old_nodes)

