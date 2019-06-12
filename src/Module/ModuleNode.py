from src.Graph.Node import Node
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


    def __init__(self, value = None):
        Node.__init__(self,value)
        self.deepLayer = nn.Linear(5,5)
        self.traversalID = ""

    def insertAggregatorNodes(self, state = "start"):#could be made more efficient as a breadth first instead of depth first because of duplicate paths
        from src.Module.AggregatorNode import AggregatorNode as Aggregator
        """
        *NB  -- must have called getTraversalIDs on root node to use this function
        traverses the module graph from the input up to output, and inserts aggregators
        which are module nodes which take multiple inputs and combine them"""

        if(state == "start"):#method is called from the root node - but must be traversed from the output node
            outputNode = self.getOutputNode()
            outputNode.insertAggregatorNodes("fromTop")
            return

        #print("inserting agg nodes from " , self.traversalID)

        numParents = len(self.parents)

        if(numParents > 1):
            #must insert an aggregator node
            aggregator = Aggregator()#all parents of self must be rerouted to this aggregatorNode
            #self.getInputNode().registerAggregator(aggregator)

            for parent in self.parents:
                aggregator.addParent(parent)
                parent.children = [aggregator if x == self else x for x in parent.children]#replaces self as a child with the new aggregator node

            self.parents = []#none of the parents are now a parent to this node - instead only the aggregator is
            #self.parents.append(aggregator)
            aggregator.addChild(self)

            for previousParent in aggregator.parents:
                previousParent.insertAggregatorNodes("fromTop")

        elif (numParents == 1):
            self.parents[0].insertAggregatorNodes("fromTop")


        if(self.isOutputNode()):
            self.getInputNode().getTraversalIDs('_')

    def passANNInputUpGraph(self, input, parentID = ""):
        """called by the forward method of the NN - traverses the module graph
            passes the nn's input all the way up through the graph
            aggregator nodes wait for all inputs before passing outputs
            """

        if(self.isOutputNode()):
            print("final node received input:", input)

        output = self.passInputThroughLayer(input)#will call aggregation if is aggregator node
        # if(input is None):
        #     print("out of agg(",type(self), "):",output)

        for child in self.children:
            # if (input is None):
            #     print("passing output to child:",child)
            child.passANNInputUpGraph(output, self.traversalID)

        return output

    def passInputThroughLayer(self, input):
        return self.deepLayer(input)

    def printNode(self, printToConsole = True):
        out = " "*(len(self.traversalID)) + self.traversalID
        if(printToConsole):
            print(out)
        else:
            return out




