import matplotlib.pyplot as plt
import math

class Node:

    """
    All children lead to the leaf node
    """

    children = []
    parents = []
    value = None

    traversalID = "" # a string structured as '1,1,3,2,0' where each number represents which child to move to along the path from input to output


    def __init__(self, val = None):
        self.value = val
        self.children = []
        self.parents = []

    def addChild(self, value = None):
        self.addChild(Node(value))

    def addChild(self, childNode):
        """
        :param childNode: Node to be added - can have subtree underneath
        """
        self.children.append(childNode)
        childNode.parents.append(self)

    def getChild(self, childNum):
        return self.children[childNum]

    def getOutputNode(self):
        if(len(self.children) == 0):
            return self

        return self.children[0].getOutputNode()

    def getInputNode(self):
        if(len(self.parents) == 0):
            return self

        return self.parents[0].getInputNode()


    def getTraversalIDs(self, currentID = ""):
        """should be called on root node
            calculates all nodes traversal ID
        """
        self.traversalID = currentID
        #print(self,"num children:", len(self.children))
        #print("Me:",self,"child:",self.children[0])
        for childNo in range(len(self.children)):
            newID = currentID + (',' if not currentID == "" else "") + repr(childNo)
            #print(newID)
            self.children[childNo].getTraversalIDs(newID)

    def isInputNode(self):
        return len(self.parents) == 0

    def isOutputNode(self):
        return len(self.children) == 0

    def printTree(self, nodesPrinted = set()):
        if (self in nodesPrinted):
            return
        nodesPrinted.add(self)
        self.printNode()

        for child in self.children:
            child.printTree(nodesPrinted)

    def printNode(self, printToConsole = True):
        pass

    def plotTree(self, nodesPlotted, rotDegree = 0):
        arrowScaleFactor = 1

        y = len(self.traversalID)
        x = 0

        for i in range(4):
            x += self.traversalID.count(repr(i)) * i

        #x +=y*0.05

        x = x* math.cos(rotDegree) - y * math.sin(rotDegree)
        y = y* math.cos(rotDegree) + x * math.sin(rotDegree)

        if (self in nodesPlotted):
            return x,y

        nodesPlotted.add(self)

        plt.plot(x, y, self.getPlotColour(),markersize=10)

        for child in self.children:
            c = child.plotTree(nodesPlotted, rotDegree)
            if(not c == None):
                cx, cy = c
                plt.arrow(x,y , (cx - x)*arrowScaleFactor , (cy - y) * 0.8*arrowScaleFactor, head_width=0.13, length_includes_head=True)

                #print("plotting from:",(x,y),"to",(cx,cy))
            #print(child.plotTree(nodesPlotted,xs,ys))

        if(self.isInputNode()):
            plt.show()

        return x , y

    def getPlotColour(self):
        return 'ro'

def genNodeGraph(nodeType, graphType = "diamond"):
    """the basic starting points of both blueprints and modules"""
    input = nodeType()

    if(graphType == "linear"):
        input.addChild(nodeType())
        input.children[0].addChild(nodeType())

    if(graphType == "diamond"):
        input.addChild(nodeType())
        input.addChild(nodeType())
        input.children[0].addChild(nodeType())
        input.children[1].addChild(input.children[0].children[0])

    if(graphType == "triangle"):
        """feeds input node to a child and straight to output node"""
        input.addChild(nodeType())
        input.children[0].addChild(nodeType())
        input.addChild(input.children[0].children[0])

    return input

