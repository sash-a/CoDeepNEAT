from src.Graph.Node import Node
import torch.nn as nn
import torch

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


    def __init__(self ):
        Node.__init__(self)
        self.deepLayer = None#an nn layer object such as    nn.Conv2d(3, 6, 5) or nn.Linear(84, 10)
        self.inFeatures = -1
        self.outFeatures = -1
        self.traversalID = ""

    def createLayers(self, inChannels = None, outChannels = 20):
        self.outFeatures = outChannels
        if(self.deepLayer is None):
            if (inChannels is None):
                self.inFeatures = self.parents[0].outFeatures  # only aggregator nodes should have more than one parent
            else:
                self.inFeatures = inChannels
            self.deepLayer = nn.Conv2d(self.inFeatures, self.outFeatures, 3, 1)


            for child in self.children:
                child.createLayers()

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

        # if(self.isOutputNode()):
        #     print("final node received input:", input)

        output = self.passInputThroughLayer(input)#will call aggregation if is aggregator node
        # if(input is None):
        #     print("out of agg(",type(self), "):",output)

        childOut = None
        for child in self.children:
            # if (input is None):
            #     print("passing output to child:",child)
            co = child.passANNInputUpGraph(output, self.traversalID)
            if(not co is None):
                childOut = co

        if(not childOut is None):
            #print("bubbling up output nodes output:", childOut.size())
            return childOut

        if(self.isOutputNode()):
            #print("output node returning output:",output)
            return output#output of final output node must be bubbled back up to the top entry point of the nn

    def passInputThroughLayer(self, input):
        #print("inshape:",input.size())
        #print("outshape:", self.deepLayer(input).size())
        return self.deepLayer(input)

    def getParameters(self, parametersDict):
        if(not self in parametersDict):
            myParams = self.deepLayer.parameters()
            parametersDict[self] = myParams

            for child in self.children:
                child.getParameters(parametersDict)

            if(self.isInputNode()):
                #print("input node params: ", parametersDict)

                params = None
                for param in parametersDict.values():

                    if(params is None):
                        params = list(param)
                    else:
                        params += list(param)
                return params

    def printNode(self, printToConsole = True):
        out = " "*(len(self.traversalID)) + self.traversalID
        if(printToConsole):
            print(out)
        else:
            return out

    def getDimensionality(self):
        print("need to implement get dimensionality")
        #print(torch.cumprod(self.deepLayer.shape()))
        return 10*10*self.deepLayer.out_channels #10*10 because by the time the 28*28 has gone through all the convs - it has been reduced to 10810




