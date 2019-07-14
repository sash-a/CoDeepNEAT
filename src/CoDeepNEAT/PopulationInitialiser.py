from src.CoDeepNEAT.CDNNodes import ModulenNEATNode
from CoDeepNEAT.CDNNodes import BlueprintNEATNode
from CoDeepNEAT.CDNGenomes import BlueprintGenome, ModuleGenome
from src.NEAT.Gene import ConnectionGene, NodeType


def initialise_blueprints():
    linear_nodes = [BlueprintNEATNode(0, node_type=NodeType.INPUT),
                    BlueprintNEATNode(1, node_type=NodeType.OUTPUT)]
    tri_nodes = [BlueprintNEATNode(0, node_type=NodeType.INPUT),
                 BlueprintNEATNode(2, node_type=NodeType.HIDDEN),
                 BlueprintNEATNode(1, node_type=NodeType.OUTPUT)]

    linear_connections = [ConnectionGene(0, linear_nodes[0].id, linear_nodes[1].id)]
    tri_connections = [ConnectionGene(0, tri_nodes[0].id, tri_nodes[2].id),
                       ConnectionGene(1, tri_nodes[0].id, tri_nodes[1].id),
                       ConnectionGene(2, tri_nodes[1].id, tri_nodes[2].id)]

    pop = [
            BlueprintGenome(linear_connections, linear_nodes),
            BlueprintGenome(tri_connections, tri_nodes)
        ]
    for indv in pop:
        indv.calculate_heights()

    return pop


def initialise_modules():
    linear_nodes = [ModulenNEATNode(0, node_type=NodeType.INPUT),
                    ModulenNEATNode(1, node_type=NodeType.OUTPUT)]
    tri_nodes = [ModulenNEATNode(0, node_type=NodeType.INPUT),
                 ModulenNEATNode(2, node_type=NodeType.HIDDEN),
                 ModulenNEATNode(1, node_type=NodeType.OUTPUT)]

    linear_connections = [ConnectionGene(0, linear_nodes[0].id, linear_nodes[1].id)]
    tri_connections = [ConnectionGene(0, tri_nodes[0].id, tri_nodes[2].id),
                       ConnectionGene(1, tri_nodes[0].id, tri_nodes[1].id),
                       ConnectionGene(2, tri_nodes[1].id, tri_nodes[2].id)]

    pop = [
        ModuleGenome(linear_connections, linear_nodes),
        ModuleGenome(tri_connections, tri_nodes)
    ]
    for indv in pop:
        indv.calculate_heights()

    return pop


def initialize_mutations():
    return {(0, 1): 0,  # linear connection
            0: 2,  # node mutation on linear connection
            (0, 2): 1,  # connection mutation for above node mutation
            (2, 1): 2}  # connection mutation for above node mutation
