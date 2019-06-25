import copy
import random

from src.NEAT.Genotype import Genome


def crossover(parent1, parent2):
    # Choosing the fittest parent
    if parent1.fitness == parent2.fitness:  # if the fitness is the same choose the shortest
        best_parent, worst_parent = (parent2, parent1) \
            if len(parent2.connections) < len(parent1.connections) else (parent1, parent2)
    else:
        best_parent, worst_parent = (parent2, parent1) \
            if parent2.fitness > parent1.fitness else (parent1, parent2)

    # disjoint + excess are inherited from the most fit parent
    d, e = copy.deepcopy(best_parent.get_disjoint_excess(worst_parent))
    child = Genome(d + e, list(set(parent2.nodes + parent1.nodes)))

    # Finding the remaining matching genes and choosing randomly between them
    for best_conn in best_parent.connections:
        if best_conn.innovation in worst_parent.innov_nums:
            worst_conn = copy.deepcopy(worst_parent.get_connection(best_conn.innovation))
            child.add_connection(random.choice([best_conn, worst_conn]))

    return child
