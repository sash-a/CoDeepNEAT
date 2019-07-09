from Test.TestingGenomes import moo_pop_members
from src.NEAT.MultiobjectivePopulation import MultiobjectivePopulation


def test_adjust_fitness():
    pop = MultiobjectivePopulation(moo_pop_members, {}, 2, 0, 0, 0)
    moo_pop_members[0].report_fitness(3, 3)
    pop.adjust_fitness(moo_pop_members[0])
    assert moo_pop_members[0].adjusted_fitness == 9

    pop.speciation_thresh = 0
    pop.adjust_fitness(moo_pop_members[0])
    assert moo_pop_members[0].adjusted_fitness == 18


def test_save_elite():
    pass
