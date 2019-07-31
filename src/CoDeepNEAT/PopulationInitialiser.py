from src.NEAT.Gene import ConnectionGene, NodeType


def initialize_pop(Node, GenomeType, initial_individuals, create_triangles=True):
    in_node_params = (0, NodeType.INPUT)
    out_node_params = (1, NodeType.OUTPUT)
    tri_node_params = (2, NodeType.HIDDEN)

    pop = []
    individuals_to_create = initial_individuals if not create_triangles else initial_individuals // 2

    for _ in range(individuals_to_create):
        pop.append(GenomeType([ConnectionGene(0, 0, 1)],
                              [Node(*in_node_params), Node(*out_node_params)]))

        if create_triangles:
            pop.append(GenomeType([ConnectionGene(0, 0, 1), ConnectionGene(1, 0, 2), ConnectionGene(2, 2, 1)],
                                  [Node(*in_node_params), Node(*tri_node_params), Node(*out_node_params)]))

    for indv in pop:
        indv.calculate_heights()

    return pop


# def initialise_blueprints():
#     linear_nodes = [BlueprintNEATNode(0, node_type=NodeType.INPUT),
#                     BlueprintNEATNode(1, node_type=NodeType.OUTPUT)]
#     tri_nodes = [BlueprintNEATNode(0, node_type=NodeType.INPUT),
#                  BlueprintNEATNode(2, node_type=NodeType.HIDDEN),
#                  BlueprintNEATNode(1, node_type=NodeType.OUTPUT)]
#
#     linear_connections = [ConnectionGene(0, linear_nodes[0].id, linear_nodes[1].id)]
#     tri_connections = [ConnectionGene(0, tri_nodes[0].id, tri_nodes[2].id),
#                        ConnectionGene(1, tri_nodes[0].id, tri_nodes[1].id),
#                        ConnectionGene(2, tri_nodes[1].id, tri_nodes[2].id)]
#
#     pop = [
#         BlueprintGenome(linear_connections, linear_nodes),
#         BlueprintGenome(tri_connections, tri_nodes)
#     ]
#     for indv in pop:
#         indv.calculate_heights()
#
#     return pop


def initialize_mutations(create_triangles=False):
    if not create_triangles:
        return {(0, 1): 0}  # linear connection

    return {(0, 1): 0,  # linear connection
            0: 2,  # node mutation on linear connection
            (0, 2): 1,  # connection mutation for above node mutation
            (2, 1): 2}  # connection mutation for above node mutation
