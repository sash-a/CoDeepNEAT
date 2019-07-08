import copy
import random

from src.NEAT.Connection import Connection


def add_nodes_from_connections(conn: Connection, genome):
    # Add the node to the genome if it is not already there
    if conn.from_node.id not in genome.node_ids:
        new_from = copy.deepcopy(conn.from_node)
        genome.add_node(new_from)
        conn.from_node = new_from
    else:
        # If the node is in the genome make the connection point to that node
        new_from = conn.from_node = genome.get_node(conn.from_node.id)

    if conn.to_node.id not in genome.node_ids:
        new_to = copy.deepcopy(conn.to_node)
        genome.add_node(new_to)
        conn.to_node = new_to
    else:
        new_to = conn.to_node = genome.get_node(conn.to_node.id)

    return new_from, new_to


def crossover(parent1, parent2):
    # Choosing the fittest parent
    if parent1.fitness == parent2.fitness:  # if the fitness is the same choose the shortest
        best_parent, worst_parent = (parent2, parent1) \
            if len(parent2.connections) < len(parent1.connections) else (parent1, parent2)
    else:
        best_parent, worst_parent = (parent2, parent1) \
            if parent2.fitness > parent1.fitness else (parent1, parent2)

    # if type(parent1) == ModuleGenome:
    #     child = ModuleGenome([], [])
    # else:
    #     child = BlueprintGenome([], [])
    if type(parent1) != type(parent2):
        raise TypeError('Parent 1 and 2 must be of the same type')

    child = type(parent1)([], [])

    # Crossing over connections
    for best_conn in best_parent.connections:
        if best_conn.innovation in worst_parent.innov_nums:  # connection in both parent choose 1
            worst_conn = worst_parent.get_connection(best_conn.innovation)
            choice = copy.deepcopy(random.choice([best_conn, worst_conn]))
            child.add_connection(choice)
        else:  # disjoint/excess
            child.add_connection(copy.deepcopy(best_conn))

    # Crossing over nodes
    for best_node in best_parent.nodes:
        if best_node.id in worst_parent.node_ids:  # node in both parent choose 1
            worst_node = worst_parent.get_node(best_node.id)
            choice = copy.deepcopy(random.choice([best_node, worst_node]))
            child.add_node(choice)
        else:  # disjoint/excess
            child.add_node(copy.deepcopy(best_node))

    return child
