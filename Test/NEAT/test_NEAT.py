import random

from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NEATNode, NodeType
from src.NEAT.Genome import Genome
from src.NEAT.Species import Species
from src.NEAT.Crossover import crossover

nodes = [NEATNode(0, 0, NodeType.INPUT),
         NEATNode(1, 0, NodeType.INPUT),
         NEATNode(2, 0, NodeType.INPUT),
         NEATNode(3, 1, NodeType.HIDDEN),
         NEATNode(4, 2, NodeType.OUTPUT)
         ]
"""
    4
   / \
  /   3
 /   /  \
2   0    1
"""
connections = \
    [
        Connection(nodes[0], nodes[3], innovation=0),
        Connection(nodes[1], nodes[3], innovation=1),
        Connection(nodes[2], nodes[4], innovation=2),
        Connection(nodes[3], nodes[4], innovation=3)
    ]


def test_midpoint():
    assert nodes[0].midpoint(nodes[1]) == 0
    assert nodes[0].midpoint(nodes[3]) == 0.5
    assert nodes[0].midpoint(nodes[4]) == 1


def test_add_connection():
    indv = Genome([connections[2]], nodes)
    indv.add_connection(connections[3])
    indv.add_connection(connections[0])
    indv.add_connection(connections[1])

    assert indv.connections == connections

    assert Genome(connections, nodes).connections == connections


def test_get_disjoint_excess():
    indv_full = Genome(connections, nodes)
    indv_single = Genome([connections[1]], nodes)

    d, e = indv_full.get_disjoint_excess(indv_single)
    assert d == [connections[0]]
    assert e == [connections[2], connections[3]]

    assert indv_single.get_disjoint_excess(indv_full) == ([], [])

    assert indv_full.get_disjoint_excess(indv_full) == ([], [])


def test_distance_to():
    full = Genome(connections, nodes)
    no_excess = Genome([connections[1], connections[3]], nodes)
    excess = Genome([connections[1], connections[2]], nodes)
    single = Genome([connections[0]], nodes)

    assert full.distance_to(no_excess) == 0.5
    assert full.distance_to(excess) == 0.5
    assert full.distance_to(single) == 0.75
    assert excess.distance_to(single) == 1.5
    assert no_excess.distance_to(single) == 1.5

    assert full.distance_to(full) == no_excess.distance_to(no_excess) == excess. \
        distance_to(excess) == single.distance_to(single) == 0


def test_is_compatible():
    spc = Species(Genome(connections, nodes))
    assert spc.is_compatible(spc.representative, 0)

    no_excess = Genome([connections[1], connections[3]], nodes)
    excess = Genome([connections[1], connections[2]], nodes)
    single = Genome([connections[0]], nodes)

    assert spc.is_compatible(no_excess, 0.5) == spc.is_compatible(excess, 0.5) is True
    assert spc.is_compatible(single, 0.75)


def test_mutation():
    # Add node
    g = Genome(connections, nodes)
    mutations = dict()

    n_innov, n_node_id = g._mutate_add_node(connections[2], mutations, 3, 4)
    expected_node = NEATNode(5, 1)
    expected_from_conn = Connection(nodes[2], expected_node)
    expected_to_conn = Connection(expected_node, nodes[4])

    assert expected_node in g.nodes
    assert expected_from_conn in g.connections
    assert expected_to_conn in g.connections
    assert not connections[2].enabled.get_value()
    assert n_innov == 5
    assert n_node_id == 5
    assert connections[2].innovation in mutations
    assert mutations[connections[2].innovation] == n_node_id

    # Add connection
    n_innov = g._mutate_add_connection(nodes[2], nodes[3], mutations, 5)
    expected_conn = Connection(nodes[2], nodes[3])
    assert expected_conn in g.connections
    assert n_innov == 6
    assert (nodes[2].id, nodes[3].id) in mutations
    assert mutations[(nodes[2].id, nodes[3].id)] == n_innov

    # Innovation numbers shouldnt increase mutation already exists
    # fake mutation
    fake_node_id = 7
    mutations[connections[0].innovation] = fake_node_id
    mutations[(connections[0].from_node.id, fake_node_id)] = connections[0].innovation
    mutations[(fake_node_id, connections[0].to_node.id)] = connections[0].innovation

    n_innov, n_node_id = g._mutate_add_node(connections[0], mutations, 8, 5)
    assert n_innov == 8
    assert n_node_id == 5


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


def test_speciation():
    pass


def test_adjust_fitness():
    pass
