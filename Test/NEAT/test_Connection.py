from src.NEAT.Connection import Connection

from Test.TestingGenomes import nodes, connections


def test_eq():
    fake_conn = Connection(nodes[0], nodes[3], innovation=3)
    assert fake_conn == connections[0]
    for conn in connections:
        assert conn == conn
