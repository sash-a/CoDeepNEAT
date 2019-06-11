from CoDeepNEAT.src.Graph.Node import Node
from CoDeepNEAT.src.Module.AggregatorNode import AggregatorNode
import torch.nn as nn


class ModuleNode(Node):
    """
    ModuleNode represents a node in a module
    The whole  Module is represented as the input Module, followed by its children
    Each Module should have one input node and one output node
    All module children are closer to the output node than their parent (one way/ feed forward)

    Modules get parsed into neural networks via traversals
    """

    layerType = None
    regularisation = None
    deepLayer = None #an nn layer object such as    nn.Conv2d(3, 6, 5) or nn.Linear(84, 10)

    traversalID = "" # a string structured as '1,1,3,2,0' where each number represents which child to move to along the path from input to output

    def __init__(self, value = None):
        Node.__init__(self,value)

    def getTraversalIDs(self, currentID = ""):
        """should be called on root node
            calculates all nodes traversal ID
        """
        self.traversalID = currentID
        for childNo in len(self.children):
            newID = (',' if not currentID == "" else "") + repr(childNo)
            self.children()[childNo].getTraversalIDs(newID)

    def insertAggregatorNodes(self, state = "start"):#could be made more efficient as a breadth first instead of depth first because of duplicate paths
        """
        *NB  -- must have called getTraversalIDs on root node to use this function
        traverses the module graph from the input up to output, and inserts aggregators
        which are module nodes which take multiple inputs and combine them"""

        if(state == "start"):#method is called from the root node - but must be traversed from the output node
            outputNode = self.getLeafNode()
            outputNode.insertAggregatorNodes("fromTop")


        numParents = len(self.parents)

        if(numParents > 1):
            #must insert an aggregator node
            aggregator = AggregatorNode()#all parents of self must be rerouted to this aggregatorNode
            self.getInputNode().registerAggregator(aggregator)

            for parent in self.parents:
                aggregator.addParent(parent)
                parent.children = [aggregator if x==self else x for x in parent.children]#replaces self as a child with the new aggregator node

            self.parents = []#none of the parents are now a parent to this node - instead only the aggregator is
            self.parents.append(aggregator)

            for previousParent in aggregator.parents:
                previousParent.insertAggregatorNodes("fromTop")

    def passANNInputUpGraph(self, input, parentID = ""):
        """called by the forward method of the NN - traverses the module graph
            passes the nn's input all the way up through the graph
            aggregator nodes wait for all inputs before passing outputs
            """

        output = self.deepLayer(input)

        for child in self.children:
            child.passANNInputUpGraph(output, self.traversalID)





