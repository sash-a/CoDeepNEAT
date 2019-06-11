import enum

class Node:

    """
    All children lead to the leaf node
    """

    children = []
    parents = []
    value = None

    def __init__(self, val):
        self.value = val

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

    def getLeafNode(self):
        if(len(self.children) == 0):
            return self

        return self.children[0].getLeafNode()

    def getInputNode(self):
        if(len(self.parents) == 0):
            return self

        return self.parents[0].getInputNode()

