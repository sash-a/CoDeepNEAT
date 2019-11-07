import os

from Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.NEAT.Genome import Genome
from test import StaticGenomes

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from graphviz import Digraph


def get_graph_of(genome: Genome, sub_graph=False, cluster_style="filled", cluster_colour="lightgrey",
                 node_style="filled", node_colour="white", label="",
                 node_shape="", start_node_shape="Mdiamond", end_node_shape="Msquare",
                 node_names="", append_graph: Digraph = None,
                 exclude_unconnected_nodes=True, exclude_non_fully_connected_nodes=True) -> Digraph:
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

        if shape != "":
            # print("using shape ",shape)
            g.node(name=node_names + "_v " + str(node.id), shape=shape, label="node " + str(node.id))
        else:
            g.node(name=node_names + "_v " + str(node.id), label="node " + str(node.id))

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


def visualise_blueprint_genome(genome: BlueprintGenome):
    pass


if __name__ == "__main__":
    genome, record = StaticGenomes.get_small_tri_genome(BlueprintGenome, BlueprintNode)
    graph = get_graph_of(genome, node_colour="yellow")
    print("genome ", genome, " parsed into graph: ", graph)
    genome.has_cycle()
    graph.view()
