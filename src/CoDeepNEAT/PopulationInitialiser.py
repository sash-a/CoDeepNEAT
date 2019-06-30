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
    blueprint_node_lists = [
        [deepcopy(blue_start), BlueprintNEATNode(1, 1)],  # no hidden
        [deepcopy(blue_start), BlueprintNEATNode(3, 1), BlueprintNEATNode(1, 2)],  # linear
        [deepcopy(blue_start), BlueprintNEATNode(4, 0), BlueprintNEATNode(1, 1)],  # triangle
        [deepcopy(blue_start), BlueprintNEATNode(5, 1), BlueprintNEATNode(6, 1), BlueprintNEATNode(1, 2)]  # diamond
    ]

    blueprint_connection_lists = [
        # no hidden
        [Connection(blueprint_node_lists[0][0], blueprint_node_lists[0][1], innovation=1)],
        # linear
        [Connection(blueprint_node_lists[1][0], blueprint_node_lists[1][1], innovation=1),
         Connection(blueprint_node_lists[1][1], blueprint_node_lists[1][2], innovation=2)],
        # triangle
        [Connection(blueprint_node_lists[2][0], blueprint_node_lists[2][1], innovation=1),
         Connection(blueprint_node_lists[2][0], blueprint_node_lists[2][2], innovation=2),
         Connection(blueprint_node_lists[2][1], blueprint_node_lists[2][2], innovation=3)],
        # diamond
        [Connection(blueprint_node_lists[3][0], blueprint_node_lists[3][1], innovation=1),
         Connection(blueprint_node_lists[3][0], blueprint_node_lists[3][2], innovation=2),
         Connection(blueprint_node_lists[3][1], blueprint_node_lists[3][3], innovation=3),
         Connection(blueprint_node_lists[3][2], blueprint_node_lists[3][3], innovation=4)],
    ]

    blueprint_reps = []
    for nodes, connections in zip(blueprint_node_lists, blueprint_connection_lists):
        blueprint_reps.append(BlueprintGenome(connections, nodes))

    return [blueprint_reps[0], blueprint_reps[1]]


def initialise_modules():
    module_node_lists = [
        [deepcopy(mod_start), ModulenNEATNode(1, 1)],  # no hidden
        [deepcopy(mod_start), ModulenNEATNode(3, 1), ModulenNEATNode(1, 2)],  # linear
        [deepcopy(mod_start), ModulenNEATNode(4, 0), ModulenNEATNode(1, 1)],  # triangle
        [deepcopy(mod_start), ModulenNEATNode(5, 1), ModulenNEATNode(6, 1), ModulenNEATNode(1, 2)]  # diamond
    ]

    module_connection_lists = [
        # no hidden
        [Connection(module_node_lists[0][0], module_node_lists[0][1], innovation=1)],
        # linear
        [Connection(module_node_lists[1][0], module_node_lists[1][1], innovation=1),
         Connection(module_node_lists[1][1], module_node_lists[1][2], innovation=2)],
        # triangle
        [Connection(module_node_lists[2][0], module_node_lists[2][1], innovation=1),
         Connection(module_node_lists[2][0], module_node_lists[2][2], innovation=2),
         Connection(module_node_lists[2][1], module_node_lists[2][2], innovation=3)],
        # diamond
        [Connection(module_node_lists[3][0], module_node_lists[3][1], innovation=1),
         Connection(module_node_lists[3][0], module_node_lists[3][2], innovation=2),
         Connection(module_node_lists[3][1], module_node_lists[3][3], innovation=3),
         Connection(module_node_lists[3][2], module_node_lists[3][3], innovation=4)],
    ]

    module_reps = []
    for nodes, connections in zip(module_node_lists, module_connection_lists):
        module_reps.append(ModuleGenome(connections, nodes))

    return module_reps
