from src.NEAT.Genome import Genome
from Test.NEAT.InitialStructure import nodes, connections, NEATNode, Connection

import random


def test_init():
    random.shuffle(connections), random.shuffle(nodes)
    g = Genome(connections, nodes)

    assert g.connections == sorted(connections, key=lambda conn: conn.innovation)
    assert g.nodes == sorted(nodes, key=lambda node: node.id)


def test_add_connection():
    g = Genome([connections[2]], nodes)
    g.add_connection(connections[3])
    g.add_connection(connections[0])
    g.add_connection(connections[1])

    assert g.connections == sorted(connections, key=lambda conn: conn.innovation)


def test_get_connection():
    g = Genome(connections, nodes)
    assert g.get_connection(10) is None

    for conn in connections:
        assert g.get_connection(conn.innovation) == conn


def test_add_node():
    g = Genome(connections, [nodes[2]])
    g.add_node(nodes[3])
    g.add_node(nodes[1])
    g.add_node(nodes[4])
    g.add_node(nodes[0])

    assert g.nodes == sorted(nodes, key=lambda node: node.id)


def test_get_node():
    g = Genome(connections, nodes)
    assert g.get_node(100) is None

    for node in nodes:
        assert g.get_node(node.id) == node


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


def test_get_disjoint_excess():
    indv_full = Genome(connections, nodes)
    indv_single = Genome([connections[1]], nodes)

    d, e = indv_full.get_disjoint_excess(indv_single)
    assert d == [connections[0]]
    assert e == [connections[2], connections[3]]

    assert indv_single.get_disjoint_excess(indv_full) == ([], [])
    assert indv_full.get_disjoint_excess(indv_full) == ([], [])


def test_mutate_add_node():
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

    # Innovation numbers shouldn't increase mutation already exists
    # fake mutation
    fake_node_id = 7
    mutations[connections[0].innovation] = fake_node_id
    mutations[(connections[0].from_node.id, fake_node_id)] = connections[0].innovation
    mutations[(fake_node_id, connections[0].to_node.id)] = connections[0].innovation

    n_innov, n_node_id = g._mutate_add_node(connections[0], mutations, 8, 5)
    assert n_innov == 8
    assert n_node_id == 5


def test_mutate_connection():
    g = Genome(connections, nodes)
    mutations = {}
    # Add connection
    n_innov = g._mutate_add_connection(nodes[2], nodes[3], mutations, 5)
    expected_conn = Connection(nodes[2], nodes[3])
    assert expected_conn in g.connections
    assert n_innov == 6
    assert (nodes[2].id, nodes[3].id) in mutations
    assert mutations[(nodes[2].id, nodes[3].id)] == n_innov

    # Innovation numbers shouldn't increase for fake mutation
    mutations[(1, 0)] = 7  # fake mutation
    n_innov = g._mutate_add_connection(nodes[0], nodes[1], mutations, 7)
    assert n_innov == 7
    assert Connection(nodes[1], nodes[0]) in g.connections

    # Making sure cyclic connection cannot be created
    fail = g._mutate_add_connection(nodes[1], nodes[0], mutations, 7)
    assert fail is None


def test_to_phenotype():
    pass
