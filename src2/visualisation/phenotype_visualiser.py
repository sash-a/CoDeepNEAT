from typing import List

import graphviz

from runs import runs_manager
from src2.configuration import config
from src2.phenotype import neural_network
from src2.phenotype.neural_network.layers.base_layer import BaseLayer


def visualise(pheno: neural_network, prefix="", suffix = ""):

    name = prefix + "blueprint_i" + str(pheno.blueprint.id) + "_phenotype" + suffix
    # print("saving:", name, "to",RunsManager.get_graphs_folder_path(config.run_name))
    graph = graphviz.Digraph(name=name, comment='phenotype')

    q: List[BaseLayer] = [pheno.model]
    graph.node(pheno.model.name, pheno.model.get_layer_info())
    visited = set()

    while q:
        parent_layer = q.pop()

        if parent_layer.name not in visited:
            visited.add(parent_layer.name)
            for child_layer in parent_layer.child_layers:
                description = child_layer.get_layer_info()

                graph.node(child_layer.name, child_layer.name + '\n' + description)
                graph.edge(parent_layer.name, child_layer.name)

                q.append(child_layer)

    try:
        graph.render(directory=runs_manager.get_graphs_folder_path(config.run_name),
                     view=config.view_graph_plots)

    except Exception as e:
        print(e)