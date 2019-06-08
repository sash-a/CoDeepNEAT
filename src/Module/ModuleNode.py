from CoDeepNEAT.src.Graph.Node import Node
from CoDeepNEAT.src.Graph.Node import Position

class Module(Node):


    def __init__(self, value = None, position = Position.input):
        Node.__init__(self,value,position)

