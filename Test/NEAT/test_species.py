from src.NEAT.Species import Species
from Test.NEAT.InitialStructure import connections, nodes, moo_pop_members
from src.NEAT.Genome import Genome

import copy


def test_add_member():
    # g = Genome(connections, nodes)
    s = Species(moo_pop_members[0])
    assert s.representative == moo_pop_members[0]
    assert s.members == [moo_pop_members[0]]

    s.add_member(copy.deepcopy(moo_pop_members[0]), thresh=0.1, safe=True)
    assert len(s.members) == 2

    s.add_member(copy.deepcopy(moo_pop_members[1]), thresh=0.1, safe=True)
    assert len(s.members) == 2

    #
    s.add_member(copy.deepcopy(moo_pop_members[1]), thresh=0.1, safe=False)
    assert len(s.members) == 3


def test_is_compatible():
    spc = Species(Genome(connections, nodes))
    assert spc.is_compatible(spc.representative, 0)

    no_excess = Genome([connections[1], connections[3]], nodes)
    excess = Genome([connections[1], connections[2]], nodes)
    single = Genome([connections[0]], nodes)

    assert spc.is_compatible(no_excess, 0.5) == spc.is_compatible(excess, 0.5) is True
    assert spc.is_compatible(single, 0.75)


def test_pareto_front():
    pass
