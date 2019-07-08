from src.NEAT.Species import Species
from Test.NEAT.InitialStructure import connections, nodes
from src.NEAT.Genome import Genome


def test_add_member():
    pass


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
