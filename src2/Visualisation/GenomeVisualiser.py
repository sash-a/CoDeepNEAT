from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union, List, Dict

from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from test import StaticGenomes

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Genome import Genome
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from graphviz import Digraph


def get_graph_of(genome: Genome, sub_graph=False, cluster_style="filled", cluster_colour="lightgrey",
                 node_style="filled", node_colour="white", label="",
                 node_shape="", start_node_shape="Mdiamond", end_node_shape="Msquare",
                 node_names="", append_graph: Digraph = None,
                 exclude_unconnected_nodes=True, exclude_non_fully_connected_nodes=True,
                 **kwargs) -> Digraph:
    """
    :param genome:
    :param sub_graph: boolean, if this graph should be made a cluster, intended to be a sub_graph of another graph
    :param cluster_style:
    :param cluster_colour:
    :param node_style:
    :param node_colour:
    :param label:
    :param node_names: the name prefix for each of the nodes in this genome
    :param append_graph:the existing graph to be drawn into, instead of into a new graph
    :param exclude_unconnected_nodes:
    :param exclude_non_fully_connected_nodes:
    :return:
    """

    if genome is None:
        raise Exception("null genome passed to grapher")

    if append_graph is None:
        g = Digraph(name=("cluster_" if sub_graph else "") + "genome_" + str(genome.id))
        # print("created graph ", g)
    else:
        g = append_graph

    if exclude_non_fully_connected_nodes:
        connection_set = set(genome.get_fully_connected_connections())
    else:
        """all connections"""
        connection_set = set(genome.connections.values())

    if exclude_unconnected_nodes:
        """all nodes which are attached to an enabled connection"""
        node_set = set([genome.nodes[connection.from_node_id] for connection in connection_set if connection.enabled()]
                       + [genome.nodes[connection.to_node_id] for connection in connection_set if connection.enabled()])
    else:
        """all nodes"""
        node_set = set(genome.nodes.values())

    for node in node_set:
        shape = start_node_shape if (node.is_input_node() and start_node_shape != "") else (
            end_node_shape if (node.is_output_node() and end_node_shape != "") else node_shape)

        if "sample_map" in kwargs and kwargs["sample_map"] is not None:
            meta_data = get_node_metadata(node, sample_map=kwargs["sample_map"])
        else:
            meta_data = get_node_metadata(node)

        if shape != "":
            # print("using shape ",shape)
            g.node(name=node_names + "_v " + str(node.id), shape=shape, label=meta_data)
        else:
            g.node(name=node_names + "_v " + str(node.id), label=meta_data)

        # print("created node: ", (node_names + ": " + str(node.id)) , " id: ", node.id )

    # print("graph after nodes added: " , g)

    for conn in connection_set:
        if not conn.enabled():
            continue

        g.edge((node_names + "_v " + str(conn.from_node_id)), (node_names + "_v " + str(conn.to_node_id)))

    # print("graph after edges added: " , g)

    if sub_graph:
        g.attr(style=cluster_style, color=cluster_colour)
        # print("changed subgraph style")
    g.node_attr.update(style=node_style, color=node_colour)

    g.attr(label=label)

    return g


def visualise_blueprint_genome(genome: BlueprintGenome, sample_map: Dict[int, int] = None):
    blueprint_graph = get_graph_of(genome, node_names="blueprint", sample_map=sample_map, node_colour="yellow")
    module_ids = set()

    for bp_node in genome.nodes.values():
        if bp_node.linked_module_id != -1:
            module_ids.add(bp_node.linked_module_id)

    if sample_map is not None:
        for module_id in sample_map.values():
            module_ids.add(module_id)

    import src2.main.Singleton as Singleton

    for module_id in module_ids:
        print("found module node")
        module = Singleton.instance.module_population[module_id]
        module_graph = get_graph_of(module, node_names="module_" + str(module_id),
                                    sub_graph=True, label="Module " + str(module_id), node_colour="blue")
        blueprint_graph.subgraph(module_graph)

    blueprint_graph.view()


def visualise_traversal_dict(traversal_dict: Dict[int, List[int]]):
    g = Digraph(name="traversal_dict")

    for from_id in traversal_dict.keys():
        g.node(name=str(from_id))
        for to_id in traversal_dict[from_id]:
            g.node(name=str(to_id))
            g.edge(str(from_id), str(to_id))

    g.view()


def get_node_metadata(node: Union[BlueprintNode, ModuleNode], **kwargs):
    meta = ""
    if isinstance(node, BlueprintNode):
        # print("found bp node")
        blueprintNode: BlueprintNode = node
        meta += "Species: " + str(blueprintNode.species_id)
        meta += "\nGene id: " + str(blueprintNode.id)

        module_id = blueprintNode.linked_module_id
        if "sample_map" in kwargs and kwargs["sample_map"] is not None:
            if blueprintNode.species_id in kwargs["sample_map"]:
                module_id = kwargs["sample_map"][blueprintNode.species_id]

        meta += "\nModule: " + str(module_id)
        if blueprintNode.module_repeat_count() > 1:
            meta += "\nRepeat count: " + str(blueprintNode.module_repeat_count())

    if isinstance(node, ModuleNode):
        # print("found module node")
        moduleNode: ModuleNode = node
        if moduleNode.is_conv():
            window_size = moduleNode.layer_type.get_subvalue("conv_window_size")
            meta += "Conv " + str(window_size) + "*" + str(window_size)
            if moduleNode.layer_type.get_subvalue("reduction") is not None:
                print('found reduction:' , moduleNode.layer_type.get_subvalue("reduction"), "pretty:", pretty(repr(moduleNode.layer_type.get_subvalue("reduction"))) )
                meta += "\nReduction: " + pretty(repr(moduleNode.layer_type.get_subvalue("reduction")))

        if moduleNode.is_linear():
            out_features = moduleNode.layer_type.get_subvalue("out_features")
            meta += "Linear " + str(out_features)

        if moduleNode.layer_type.get_subvalue("regularisation") is not None:
            print("found reg",moduleNode.layer_type.get_subvalue("regularisation"),"pretty:",pretty(repr(moduleNode.layer_type.get_subvalue("regularisation"))))
            meta += "\nRegularisation: " + pretty(repr(moduleNode.layer_type.get_subvalue("regularisation")))

        if moduleNode.layer_type.get_subvalue("dropout") is not None:
            fac = moduleNode.layer_type.get_submutagen("dropout").get_subvalue("dropout_factor")
            meta += "\nDropout: " + pretty(repr(moduleNode.layer_type.get_subvalue("dropout"))) + " p = " + repr(fac)

        if len(meta) == 0:
            """is identiy node"""
            meta += "Identity"

    print("labeling node", meta)

    return meta


def pretty(full_object_name: str):
    if "." not in full_object_name or "'" not in full_object_name:
        return full_object_name
    return full_object_name.split(".")[-1].split("'")[0]


if __name__ == "__main__":
    # genome, record = StaticGenomes.get_small_tri_genome(BlueprintGenome, BlueprintNode)
    # graph = get_graph_of(genome, node_colour="yellow")
    # print("genome ", genome, " parsed into graph: ", graph)
    # genome.has_cycle()
    # graph.view()
    # visualise_traversal_dict(genome.get_traversal_dictionary())

    pass
