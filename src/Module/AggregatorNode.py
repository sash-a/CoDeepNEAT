from src.Module.ModuleNode import ModuleNode as Module

class AggregatorNode(Module):

    aggregationType = ""
    moduleNodeInputIDs = []# a list of the ID's of all the modules which pass their output to this aggregator
    accountedForInputIDs = {}#map from ID's to input vectors

    def __init__(self):
        Module.__init__(self)
        self.moduleNodeInputIDs = []
        self.accountedForInputIDs = {}
        #print("created aggregator node")

    def addInputNodeID(self, id):
        self.moduleNodeInputIDs.append(id)

    def insertAggregatorNodes(self, state="start"):
        #if aggregator node has already been created - then the multi parent situation has already been dealt with here
        #since at the time the decendants of this aggregator node were travered further already, there is no need to traverse its decendants again
        pass

    def addParent(self, parent):
        self.addInputNodeID(parent.traversalID)
        self.parents.append(parent)

    def resetNode(self):
        """typically called between uses of the forward method of the NN created
            tells the aggregator a new pass is underway and all inputs must again be waited for to pass forward
        """
        self.accountedForInputIDs = {}

    def passANNInputUpGraph(self, input, parentID = ""):
        self.accountedForInputIDs[parentID] = input

        if(len(self.moduleNodeInputIDs) == len(self.accountedForInputIDs)):
            #all inputs have arrived
            #may now aggregate and pass upwards

            super(AggregatorNode, self).passANNInputUpGraph(None)
            
            self.resetNode()

    def passInputThroughLayer(self, _):
        print("aggregate inputs not yet implemented fully")
        output = None
        for id in self.moduleNodeInputIDs:
            input = self.accountedForInputIDs[id]
            if(output == None):
                output = input
            else:
                output = output + input
            #combine inputs

        return output

    def getPlotColour(self):
        #print("plotting agg node")
        return 'bo'

