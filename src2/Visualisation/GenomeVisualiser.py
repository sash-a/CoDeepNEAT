import os

from src2.Genotype.NEAT.Genome import Genome

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from graphviz import Digraph


def get_graph_of(genome: Genome, sub_graph = True, cluster_style ="filled", cluster_colour = "lightgrey",
                 node_style = "filled", node_colour = "white", label ="",
                 node_shape = "", start_node_shape = "Mdiamond", end_node_shape = "Msquare" ):
    g = Digraph(name = ("cluster_" if sub_graph else "") + "genome_" + str(genome.id))

    for node in genome.nodes.values():
        shape = start_node_shape if (node.is_input_node() and start_node_shape!="") else (end_node_shape if (node.is_output_node() and end_node_shape!="") else node_shape)
        if shape != "":
            g.node(name = str(node.id), shape = shape)
        else:
            g.node(name = str(node.id))

    for conn in genome.connections.values():
        g.edge(str(conn.from_node_id), str(conn.to_node_id))

    if sub_graph:
        g.attr(style=cluster_style, color=cluster_colour)
    g.node_attr.update(style=node_style, color=node_colour)

    g.attr(label = label)

    return g



