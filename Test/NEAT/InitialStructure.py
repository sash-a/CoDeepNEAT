from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NEATNode, NodeType
from src.CoDeepNEAT.PopulationInitialiser import initialise_blueprints

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

moo_pop = initialise_blueprints()
