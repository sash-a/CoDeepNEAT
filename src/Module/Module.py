from CoDeepNEAT.src.Module.ModuleNode import ModuleNode

class Module(ModuleNode):
    """a special case of module node
        the root node/ input node of the whole module node graph
    """

    aggregatorNodes = []

    def registerAggregator(self, aggregatorNode):
        self.aggregatorNodes.append(aggregatorNode)

