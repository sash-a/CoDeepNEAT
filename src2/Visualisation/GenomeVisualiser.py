import os

from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.NEAT.Genome import Genome

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from graphviz import Digraph


def get_graph_of(genome: Genome, sub_graph=False, cluster_style="filled", cluster_colour="lightgrey",
                 node_style="filled", node_colour="white", label="",
                 node_shape="", start_node_shape="Mdiamond", end_node_shape="Msquare",
                 node_names= "", append_graph: Digraph = None) -> Digraph:

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
    :return:
    """

    if append_graph == "":
        g = Digraph(name=("cluster_" if sub_graph else "") + "genome_" + str(genome.id))
    else:
        g = append_graph

    for node in genome.nodes.values():
        shape = start_node_shape if (node.is_input_node() and start_node_shape != "") else (
            end_node_shape if (node.is_output_node() and end_node_shape != "") else node_shape)
        if shape != "":
            g.node(name=node_names + ": " + str(node.id), shape=shape)
        else:
            g.node(name=node_names + ": " +str(node.id))

    for conn in genome.connections.values():
        g.edge(node_names + ": " +str(conn.from_node_id), node_names + ": " +str(conn.to_node_id))

    if sub_graph:
        g.attr(style=cluster_style, color=cluster_colour)
    g.node_attr.update(style=node_style, color=node_colour)

    g.attr(label=label)

    return g


def visualise_blueprint_genome(genome: BlueprintGenome):
    pass
