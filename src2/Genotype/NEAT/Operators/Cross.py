# This is the crossover method
import copy
import random

from src2.Genotype.NEAT.Genome import Genome


def over(genome_a: Genome, genome_b: Genome) -> Genome:
    """performs neat style cross over between a and other as b"""
    best = genome_a if genome_a < genome_a else genome_b
    worst = genome_a if genome_a > genome_b else genome_b

    child = type(best)([], [])

    for best_node in best.nodes.values():
        if best_node.id in worst.nodes:
            child_node = copy.deepcopy(random.choice([best_node, worst.nodes[best_node.id]]))
        else:
            child_node = copy.deepcopy(best_node)

        child.add_node(child_node)

    for best_conn in best.connections.values():
        if best_conn.id in worst.connections:
            new_connection = copy.deepcopy(random.choice([best_conn, worst.connections[best_conn.id]]))
        else:
            new_connection = copy.deepcopy(best_conn)

        child.add_connection(new_connection)  # child heights not meaningful at this stage

    child.inherit(best)
    child.parents = [best.id,worst.id]

    return child
