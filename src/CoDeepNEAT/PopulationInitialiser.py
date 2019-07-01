from src.CoDeepNEAT.ModuleNEATNode import ModulenNEATNode
from src.CoDeepNEAT.BlueprintNEATNode import BlueprintNEATNode
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome
from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NodeType

from copy import deepcopy

mod_start = ModulenNEATNode(0, 0, node_type=NodeType.INPUT)
blue_start = BlueprintNEATNode(0, 0, node_type=NodeType.INPUT)


def initialise_blueprints():
    nodes = [BlueprintNEATNode(0, 0, node_type=NodeType.INPUT),
             BlueprintNEATNode(1, 1, node_type=NodeType.OUTPUT)]  # no hidden

    # no hidden
    connections = [Connection(nodes[0], nodes[1], innovation=0)]

    return \
        [
            BlueprintGenome(deepcopy(connections), deepcopy(nodes)),
            BlueprintGenome(deepcopy(connections), deepcopy(nodes))
        ]


def initialise_modules():
    nodes = [ModulenNEATNode(0, 0, node_type=NodeType.INPUT),
             ModulenNEATNode(1, 1, node_type=NodeType.OUTPUT)]  # no hidden

    # no hidden
    connections = [Connection(nodes[0], nodes[1], innovation=0)]

    return \
        [
            ModuleGenome(deepcopy(connections), deepcopy(nodes)),
            ModuleGenome(deepcopy(connections), deepcopy(nodes))
        ]


def get_mutations():
    return {(0, 1): 1}


def test():
    nodes = [BlueprintNEATNode(0, 0, node_type=NodeType.INPUT),
             BlueprintNEATNode(1, 1, node_type=NodeType.OUTPUT),
             BlueprintNEATNode(2, 0)]

    mutations = {}
    node_id = 2
    innov = 0
    initial = BlueprintGenome([Connection(nodes[0], nodes[1], innovation=0)], deepcopy(nodes))
    initial.to_blueprint().plot_tree()

    linear = deepcopy(initial)
    innov, node_id = linear._mutate_add_node(linear.connections[0], mutations, innov, node_id)

    tri = deepcopy(initial)
    innov = tri._mutate_add_connection(tri.nodes[0], tri.nodes[2], mutations, innov)
    tri.to_blueprint().plot_tree()
    linear.to_blueprint().plot_tree()
    for conn in linear.connections:
        print(conn, conn.enabled)

# test()
