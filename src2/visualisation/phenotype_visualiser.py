from typing import List, Any

import graphviz

from runs import runs_manager
from src2.configuration import config
from src2.phenotype import neural_network
from src2.phenotype.neural_network.layers.aggregation_layer import AggregationLayer
from src2.phenotype.neural_network.layers.base_layer import BaseLayer
from src2.phenotype.neural_network.layers.layer import Layer


def visualise(pheno: neural_network, prefix="", suffix = "", node_colour: Any = "white"):

    name = prefix + "blueprint_i" + str(pheno.blueprint.id) + "_phenotype" + suffix
    # print("saving:", name, "to",RunsManager.get_graphs_folder_path(config.run_name))
    graph = graphviz.Digraph(name=name, comment='phenotype')

    q: List[BaseLayer] = [pheno.model]
    _node_colour = node_colour if isinstance(node_colour, str) else node_colour(pheno.model)
    graph.node(pheno.model.name, pheno.model.get_layer_info(), fillcolor = _node_colour, style="filled")
    visited = set()

    while q:
        parent_layer = q.pop()

        if parent_layer.name not in visited:
            visited.add(parent_layer.name)
            for child_layer in parent_layer.child_layers:
                _node_colour = node_colour if isinstance(node_colour, str) else node_colour(child_layer)
                description = child_layer.get_layer_info()

                graph.node(child_layer.name, child_layer.name + '\n' + description, fillcolor = _node_colour, style="filled")
                graph.edge(parent_layer.name, child_layer.name)

                q.append(child_layer)

    try:
        graph.render(directory=runs_manager.get_graphs_folder_path(config.run_name),
                     view=config.view_graph_plots)

    except Exception as e:
        print(e)

def get_node_colour(layer: BaseLayer) -> str:
    if isinstance(layer, AggregationLayer):
        return "violet"

    if isinstance(layer, Layer):
        if "Identity" in layer.get_layer_info():
            return "white"
        if layer.module_node.is_conv():
            return "yellow"
        if layer.module_node.is_linear():
            return "lightblue"

    return "green"
