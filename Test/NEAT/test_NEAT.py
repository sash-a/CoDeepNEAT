import random

from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NEATNode, NodeType
from src.NEAT.Genotype import Genome
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
    current_gen_mutations = set()

    n_innov, n_node_id = g._mutate_add_node(connections[2], current_gen_mutations, 3, 4)
    expected_node = NEATNode(5, 1)
    expected_from_conn = Connection(nodes[2], expected_node)
    expected_to_conn = Connection(expected_node, nodes[4])

    assert expected_node in g.nodes
    assert expected_from_conn in g.connections
    assert expected_to_conn in g.connections
    assert not connections[2].enabled
    assert n_innov == 5
    assert n_node_id == 5

    # Add connection
    n_innov = g._mutate_add_connection(nodes[2], nodes[3], current_gen_mutations, 5)
    expected_conn = Connection(nodes[2], nodes[3])
    assert expected_conn in g.connections
    assert n_innov == 6

    # Innovation numbers shouldnt increase mutation already exists
    fake_node = NEATNode(6, 0.5)
    fake_connection_from = Connection(nodes[0], fake_node, innovation=7)
    fake_connection_to = Connection(fake_node, nodes[3], innovation=8)
    fake_mutation = NodeMutation(fake_node.id, fake_connection_from, fake_connection_to)

    current_gen_mutations.add(fake_mutation)

    n_innov, n_node_id = g._mutate_add_node(connections[0], current_gen_mutations, 8, 5)
    assert n_innov == 8
    assert n_node_id == 5
    assert fake_node in g.nodes
    assert fake_connection_to in g.connections
    assert fake_connection_from in g.connections


def test_crossover():
    # g1 and g2 have no overlapping genes
    g1 = Genome([connections[1], connections[3]], [nodes[1], nodes[3], nodes[4]])
    g2 = Genome([connections[0], connections[2]], [nodes[0], nodes[2], nodes[3], nodes[4]])
    g1.fitness = 10
    g2.fitness = 1

    child = crossover(g1, g2)
    assert child.connections == g1.connections
    assert child.connections != g2.connections
    for child_node, g1_node in zip(child.nodes, g1.nodes):
        assert child_node.id == g1_node.id

    assert len(child.nodes) == 3

    g3 = Genome(connections, nodes[:5])
    g3.fitness = 10
    child = crossover(g3, g2)
    assert child.connections == connections
    assert len(child.nodes) == 5

    g2.fitness = 11
    child = crossover(g3, g2)
    assert child.connections == g2.connections

    # Cannot test randomness


def test_mutation_hashing():
    # Connection mutation
    c1 = ConnectionMutation(connections[0])
    c1_fake = ConnectionMutation(Connection(nodes[0], nodes[3], innovation=150))

    c2 = ConnectionMutation(connections[1])

    # Node Mutation
    n1 = NodeMutation(nodes[3], connections[0], connections[3])
    n1_fake = NodeMutation(nodes[3],
                           Connection(nodes[0], nodes[3], innovation=11),
                           Connection(nodes[3], nodes[4], innovation=10))

    n2 = NodeMutation(nodes[3], connections[1], connections[3])

    s = set()
    s.add(c1)
    s.add(n1)

    assert c1 in s
    assert c1_fake in s
    assert c2 not in s

    assert n1 in s
    assert n1_fake in s
    assert n2 not in s
