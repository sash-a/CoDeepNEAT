from src.CoDeepNEAT.ModuleNEATNode import ModulenNEATNode
from src.CoDeepNEAT.BlueprintNEATNode import BlueprintNEATNode
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome
from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NodeType


def initialise_blueprints():
    linear_nodes = [BlueprintNEATNode(0, 0, node_type=NodeType.INPUT),
                    BlueprintNEATNode(1, 1, node_type=NodeType.OUTPUT)]
    tri_nodes = [BlueprintNEATNode(0, 0, node_type=NodeType.INPUT),
                 BlueprintNEATNode(2, 0, node_type=NodeType.HIDDEN),
                 BlueprintNEATNode(1, 1, node_type=NodeType.OUTPUT)]

    linear_connections = [Connection(linear_nodes[0], linear_nodes[1], innovation=0)]
    tri_connections = [Connection(tri_nodes[0], tri_nodes[2], innovation=0),
                       Connection(tri_nodes[0], tri_nodes[1], innovation=1),
                       Connection(tri_nodes[1], tri_nodes[2], innovation=2)]

    return \
        [
            BlueprintGenome(linear_connections, linear_nodes),
            BlueprintGenome(tri_connections, tri_nodes)
        ]


def initialise_modules():
    linear_nodes = [ModulenNEATNode(0, 0, node_type=NodeType.INPUT),
                    ModulenNEATNode(1, 1, node_type=NodeType.OUTPUT)]
    tri_nodes = [ModulenNEATNode(0, 0, node_type=NodeType.INPUT),
                 ModulenNEATNode(2, 0, node_type=NodeType.HIDDEN),
                 ModulenNEATNode(1, 1, node_type=NodeType.OUTPUT)]

    linear_connections = [Connection(linear_nodes[0], linear_nodes[1], innovation=0)]
    tri_connections = [Connection(tri_nodes[0], tri_nodes[2], innovation=0),
                       Connection(tri_nodes[0], tri_nodes[1], innovation=1),
                       Connection(tri_nodes[1], tri_nodes[2], innovation=2)]
    return \
        [
            ModuleGenome(linear_connections, linear_nodes),
            ModuleGenome(tri_connections, tri_nodes)
        ]


def initialize_mutations():
    return {(0, 1): 0,  # linear connection
            0: 2,  # node mutation on linear connection
            (0, 2): 1,  # connection mutation for above node mutation
            (2, 1): 2}  # connection mutation for above node mutation
