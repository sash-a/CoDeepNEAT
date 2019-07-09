from src.NEAT.Crossover import crossover
from src.NEAT.Genome import Genome
from Test.TestingGenomes import nodes, connections


def test_crossover():
    # g1 and g2 have no overlapping genes
    g1 = Genome([connections[1], connections[3]], [nodes[1], nodes[3], nodes[4]])
    g2 = Genome([connections[0], connections[2]], [nodes[0], nodes[2], nodes[3], nodes[4]])
    g1.fitness = 10
    g2.fitness = 1

    child = crossover(g1, g2)
    assert child.connections == g1.connections
    assert child.connections != g2.connections
    # Expect genes to only come from the fittest parent
    for child_node, g1_node in zip(child.nodes, g1.nodes):
        assert child_node == g1_node

    assert len(child.nodes) == 3

    g3 = Genome(connections, nodes[:5])
    g3.fitness = 10
    child = crossover(g3, g2)
    assert child.connections == connections
    for child_node, g3_node in zip(child.nodes, g3.nodes):
        assert child_node == g3_node

    g2.fitness = 11
    child = crossover(g3, g2)
    assert child.connections == g2.connections

    # Cannot test randomness
