import random

import pytest
from src.NEAT.Connection import Connection
from src.NEAT.Node import Node, NodeType
from src.NEAT.Genotype import Genome
from src.NEAT.Species import Species

nodes = [Node(0, NodeType.INPUT),
         Node(1, NodeType.INPUT),
         Node(2, NodeType.INPUT),
         Node(3, NodeType.HIDDEN),
         Node(4, NodeType.OUTPUT)
         ]
"""
    4
   / \
  /   3
 2   /  \
    0    1
"""
connections = \
    [
        Connection(nodes[0], nodes[3], innovation=0),
        Connection(nodes[1], nodes[3], innovation=1),
        Connection(nodes[2], nodes[4], innovation=2),
        Connection(nodes[3], nodes[4], innovation=3)
    ]


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


def test_is_compatible():
    spc = Species(Genome(connections, nodes))
    assert spc.is_compatible(spc.representative, 0)

    no_excess = Genome([connections[1], connections[3]], nodes)
    excess = Genome([connections[1], connections[2]], nodes)
    single = Genome([connections[0]], nodes)

    assert spc.is_compatible(no_excess, 0.5) == spc.is_compatible(excess, 0.5) is True
    assert spc.is_compatible(single, 0.75)
