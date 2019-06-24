import random

import pytest
from src.NEAT.NEAT import NeatNode, NodeType, NeatConnection, NeatIndividual, NeatSpecies

nodes = [NeatNode(0, NodeType.INPUT),
         NeatNode(1, NodeType.INPUT),
         NeatNode(2, NodeType.INPUT),
         NeatNode(3, NodeType.HIDDEN),
         NeatNode(4, NodeType.OUTPUT)
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
        NeatConnection(nodes[0], nodes[3], innovation=0),
        NeatConnection(nodes[1], nodes[3], innovation=1),
        NeatConnection(nodes[2], nodes[4], innovation=2),
        NeatConnection(nodes[3], nodes[4], innovation=3)
    ]


def test_add_connection():
    indv = NeatIndividual([connections[2]])
    indv.add_connection(connections[3])
    indv.add_connection(connections[0])
    indv.add_connection(connections[1])

    assert indv.connections == connections

    assert NeatIndividual(connections).connections == connections


def test_max_innov():
    assert NeatIndividual(connections).get_max_innov() == 3


def test_get_disjoint_excess():
    indv_full = NeatIndividual(connections)
    indv_single = NeatIndividual([connections[1]])

    d, e = indv_full.get_disjoint_excess(indv_single)
    assert d == [connections[0]]
    assert e == [connections[2], connections[3]]

    assert indv_single.get_disjoint_excess(indv_full) == ([], [])

    assert indv_full.get_disjoint_excess(indv_full) == ([], [])


def test_is_compatible():
    spc = NeatSpecies(NeatIndividual(connections))
    assert spc.is_compatible(spc.representative, 0)

    no_excess = NeatIndividual([connections[1], connections[3]])
    excess = NeatIndividual([connections[1], connections[2]])
    single = NeatIndividual([connections[0]])

    assert spc.is_compatible(no_excess, 0.5) == spc.is_compatible(excess, 0.5) is True
    assert spc.is_compatible(single, 0.75)
