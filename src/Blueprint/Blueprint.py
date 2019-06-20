from src.Graph.Node import Node


class BlueprintNode(Node):
    """
    Each value in a blueprint graph is a Module Species number
    """

    speciesIndexesUsed = []

    def __init__(self):
        Node.__init__(self)
        self.value = 0
        self.speciesIndexesUsed = []

    def parseto_module(self, generation, module_construct=None, species_indexes=None):
        """
        :param module_construct: the output module node to have this newly sampled module attached to. None if this is
        root blueprint node
        :param generation:
        :param species_indexes:
        :return: input module node
        """
        # to be added as child to existing module construct
        input_module_node, index = generation.speciesCollection[self.value].sample_module()
        output_module_node = input_module_node.get_output_node()  # many branching modules may be added to this module

        if module_construct is not None:
            module_construct.add_child(input_module_node)
        else:
            if not self.is_input_node():
                print("null module construct passed to non root blueprint node")

        if self.is_input_node():
            self.speciesIndexesUsed = []
            species_indexes = self.speciesIndexesUsed
            species_indexes.append(index)

        # passes species index down to collect all species indexes used to construct this blueprint in one list
        for childBlueprintNode in self.children:
            childBlueprintNode.parseto_module(generation, output_module_node, species_indexes)

        if len(self.parents) == 0:
            input_module_node.get_traversal_ids("_")
            return input_module_node
