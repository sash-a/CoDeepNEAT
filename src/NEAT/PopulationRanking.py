import math
import operator
import sys

from src.Config import Config

"""a collection of the population ranking functions which were tested"""

def single_objective_rank(individuals):
    individuals.sort(key=lambda indv: (0 if not indv.fitness_values else indv.fitness_values[0]), reverse=True)
    for i, individual in enumerate(individuals):
        individual.rank = i + 1


def cdn_pareto_front(individuals):
    individuals.sort(key=lambda indv: indv.fitness_values[0], reverse=True)

    pf = [individuals[0]]  # pareto front populated with best individual in primary objective

    for indv in individuals[1:]:
        if Config.second_objective_comparator(indv.fitness_values[1], pf[-1].fitness_values[1]):
            pf.append(indv)

    return pf


def cdn_rank(individuals):
    ranked_individuals = []
    fronts = []
    remaining_individuals = set(individuals)

    while len(remaining_individuals) > 0:
        pf = cdn_pareto_front(list(remaining_individuals))
        fronts.append(pf)
        remaining_individuals = remaining_individuals - set(pf)
        ranked_individuals.extend(pf)

    for i, indv in enumerate(ranked_individuals):
        indv.rank = i + 1
    return fronts


def nsga_rank(individuals):
    fronts = general_pareto_sorting(individuals)

    rank = 1

    for front in fronts:
        # rank is firstly based on which front the indv is in
        distances = {}
        for objective in range(len(individuals[0].fitness_values)):
            # estimate density by averaging the two nearest along each objective axis, then combining each distance
            objective_sorted = sorted(front, key=lambda x: x.fitness_values[objective])
            for i, indv in enumerate(objective_sorted):
                if i == 0 or i == len(objective_sorted) - 1:
                    distance = sys.maxsize
                else:
                    distance = (abs(
                        objective_sorted[i].fitness_values[objective] - objective_sorted[i + 1].fitness_values[
                            objective]) + abs(
                        objective_sorted[i].fitness_values[objective] - objective_sorted[i - 1].fitness_values[
                            objective])) / 2
                    distance = math.pow(distance, 2)

                if objective == 0:
                    distances[indv] = []
                distances[indv].append(distance)

        distance_sorted = sorted(front, key=lambda x: sum(distances[x]), reverse=True)
        for indv in distance_sorted:
            indv.rank = rank
            rank += 1


def general_pareto_sorting(individuals, return_pareto_front_only=False):
    """takes in a list of individuals and returns a list of fronts, each being a list of individuals"""
    fronts = [[]]
    dominations = {}
    domination_counts = {}
    for indv in individuals:
        dominated_count = 0
        domination_by_indv = []
        for comparitor in individuals:
            if indv == comparitor:
                continue
            if check_domination(indv, comparitor):
                domination_by_indv.append(comparitor)
            elif check_domination(comparitor, indv):
                dominated_count += 1
        if dominated_count == 0:
            fronts[0].append(indv)

        dominations[indv] = domination_by_indv
        domination_counts[indv] = dominated_count

    if return_pareto_front_only:
        return fronts[0]

    front_number = 0
    while True:
        next_front = set()
        for leader in fronts[front_number]:
            for dominated_individual in dominations[leader]:
                domination_counts[dominated_individual] -= 1
                if domination_counts[dominated_individual] == 0:
                    next_front.add(dominated_individual)

        if len(next_front) == 0:
            break
        fronts.append(next_front)
        front_number += 1

    return fronts


def check_domination(domination_candidate, comparitor):
    """checks if the domination candidate dominates the comparitor"""
    for i in range(len(domination_candidate.fitness_values)):
        if i == 0:
            comparison = operator.gt  # objective 0 is always maximised
        else:
            comparison = Config.second_objective_comparator if i == 1 else Config.third_objective_comparator

        if comparison(comparitor.fitness_values[i], domination_candidate.fitness_values[i]):
            return False
    return True


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.NEAT.Genome import Genome
    import random

    fake_individuals = []
    for i in range(100):
        fake_individuals.append(Genome([], []))
        fake_individuals[-1].fitness_values = [random.random(), random.random()]
    fronts = general_pareto_sorting(fake_individuals)
    # print(len(fronts))
    for front in fronts:
        sorted_front = sorted(front, key=lambda indv: indv.fitness_values[0])
        x = [indv.fitness_values[0] for indv in sorted_front]
        y = [indv.fitness_values[1] for indv in sorted_front]
        plt.plot(x, y)
    plt.show()
