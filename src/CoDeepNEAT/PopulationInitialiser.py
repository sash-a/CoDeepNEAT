from src.NEAT.Gene import ConnectionGene, NodeType


def initialize_pop(Node, Genome, initial_individuals, create_triangles=True):
    in_node_params = (0, NodeType.INPUT)
    out_node_params = (1, NodeType.OUTPUT)
    tri_node_params = (2, NodeType.HIDDEN)

    pop = []
    individuals_to_create = initial_individuals if not create_triangles else initial_individuals // 2

    for _ in range(individuals_to_create):
        pop.append(Genome([ConnectionGene(0, 0, 1)],
                          [Node(*in_node_params), Node(*out_node_params)]))

        if create_triangles:
            pop.append(Genome([ConnectionGene(0, 0, 1), ConnectionGene(1, 0, 2), ConnectionGene(2, 2, 1)],
                              [Node(*in_node_params), Node(*tri_node_params), Node(*out_node_params)]))

    for indv in pop:
        indv.calculate_heights()

    return pop


def initialize_mutations(create_triangles=False):
    if not create_triangles:
        return {(0, 1): 0}  # linear connection

    return {(0, 1): 0,  # linear connection
            0: 2,  # node mutation on linear connection
            (0, 2): 1,  # connection mutation for above node mutation
            (2, 1): 2}  # connection mutation for above node mutation
