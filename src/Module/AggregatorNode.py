from CoDeepNEAT.src.Module.ModuleNode import ModuleNode

class AggregatorNode(ModuleNode):

    aggregationType = ""
    moduleNodeInputIDs = []# a list of the ID's of all the modules which pass their output to this aggregator

    def addInputNodeID(self, id):
        self.moduleNodeInputIDs.append(id)

    def insertAggregatorNodes(self, state="start"):
        #if aggregator node has already been created - then the multi parent situation has already been dealt with here
        #since at the time the decendants of this aggregator node were travered further already, there is no need to traverse its decendants again
        pass

    def addParent(self, parent):
        self.addInputNodeID(parent.traversalID)
        self.parents.append(parent)
