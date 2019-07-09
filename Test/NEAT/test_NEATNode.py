from Test.TestingGenomes import nodes


def test_midpoint():
    assert nodes[0].midpoint(nodes[1]) == 0
    assert nodes[0].midpoint(nodes[3]) == 0.5
    assert nodes[0].midpoint(nodes[4]) == 1
