from src.Module.ModuleNode import ModuleNode as Module
from src.Learner.Layers import MergeSum
import torch

class AggregatorNode(Module):

    aggregationType = ""
    moduleNodeInputIDs = []# a list of the ID's of all the modules which pass their output to this aggregator
    accountedForInputIDs = {}#map from ID's to input vectors

    def __init__(self):
        Module.__init__(self)
        self.moduleNodeInputIDs = []
        self.accountedForInputIDs = {}
        #print("created aggregator node")

    def insertAggregatorNodes(self, state="start"):
        #if aggregator node has already been created - then the multi parent situation has already been dealt with here
        #since at the time the decendants of this aggregator node were travered further already, there is no need to traverse its decendants again
        pass

    def addParent(self, parent):
        self.moduleNodeInputIDs.append(parent)
        self.parents.append(parent)

    def resetNode(self):
        """typically called between uses of the forward method of the NN created
            tells the aggregator a new pass is underway and all inputs must again be waited for to pass forward
        """
        self.accountedForInputIDs = {}

    def passANNInputUpGraph(self, input, parentID = ""):
        self.accountedForInputIDs[parentID] = input

        #print("agg(",self.traversalID,") received in:",input, "from",parentID)

        if(len(self.moduleNodeInputIDs) == len(self.accountedForInputIDs)):
            #all inputs have arrived
            #may now aggregate and pass upwards

            #print("agg(",self.traversalID,") received all inputs",self.accountedForInputIDs)

            super(AggregatorNode, self).passANNInputUpGraph(None)
            
            self.resetNode()

    def passInputThroughLayer(self, _):
        #print("aggregate inputs not yet implemented fully")
        output = None
        inputs = []
        for parent in self.moduleNodeInputIDs:
            input = self.accountedForInputIDs[parent.traversalID]

            inputs.append(input)
            #combine inputs

        #print("in:", inputs)
        #print("stack:", torch.stack(inputs))
        output = torch.sum(torch.stack(inputs), dim=0)
        #print("out:", output)

        return output

    def getPlotColour(self):
        #print("plotting agg node")
        return 'bo'

