from src.Multiobjective.ParetoFront import CDN_pareto
from Test.NEAT.InitialStructure import moo_pop_members

import random
import copy


def init():
    """Creates individuals with fitnesses: 7,1  6,2  5,3  4,4  3,5  2,6  1,7  0,8"""
    indvs = moo_pop_members
    # inflating number of individuals
    for _ in range(2):
        indvs.extend(copy.deepcopy(moo_pop_members))

    recorded_fitnesses = []
    # Setting their fitnesses
    for i, indv in enumerate(indvs, 0):
        indv.report_fitness(i, len(indvs) - i)
        recorded_fitnesses.append([i, len(indvs) - i])

    recorded_fitnesses.sort(key=lambda x: x[0], reverse=True)

    return indvs, recorded_fitnesses


def test_CDN_basic():
    indvs, recorded_fitnesses = init()

    random.shuffle(indvs)
    sorted_indvs = CDN_pareto(indvs)
    for indv, fit in zip(sorted_indvs, recorded_fitnesses):
        assert indv.fitness == fit


def test_CDN_dominated():
    indvs, recorded_fitnesses = init()
    # Making sure dominated member is not added
    excluded_mem = copy.deepcopy(moo_pop_members[0])
    excluded_mem.report_fitness(6, 0)
    indvs.append(excluded_mem)

    sorted_indvs = CDN_pareto(indvs)
    fitnesses = [indv.fitness for indv in sorted_indvs]
    assert excluded_mem.fitness not in fitnesses  # Dominated member is not on pareto front
    for indv, fit in zip(sorted_indvs, recorded_fitnesses):
        assert indv.fitness == fit  # All other members are on pareto front


def test_CDN_dominant():
    indvs, _ = init()

    # Making sure new dominant member removes all dominated members
    dominant_mem = copy.deepcopy(moo_pop_members[0])
    dominant_mem.report_fitness(6.9, 7.9)

    indvs.append(dominant_mem)
    sorted_indvs = CDN_pareto(indvs)
    fitnesses = [indv.fitness for indv in sorted_indvs]
    # Dominant member is in pareto front
    assert dominant_mem.fitness in fitnesses

    # Dominant member dominates all but 3 members
    assert fitnesses[0] == [7, 1]
    assert fitnesses[1] == [6.9, 7.9]
    assert fitnesses[2] == [0, 8]
    assert len(fitnesses) == 3
