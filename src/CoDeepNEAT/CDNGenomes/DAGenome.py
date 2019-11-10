from src.DataAugmentation.AugmentationScheme import AugmentationScheme
from src.NEAT.Genome import Genome


class DAGenome(Genome):
    """the DA variation of the genome class."""

    def __init__(self, connections, nodes):
        super().__init__(connections, nodes)

    def __repr__(self):
        node_names = []
        for node in self._nodes.values():
            if node.enabled():
                node_names.append(node.get_node_name())

        toString = "\tNodes:" + repr(list(node_names)) + "\n" + "\tTraversal_Dict: " + repr(
            self.get_traversal_dictionary())
        return "\n" + "\tConnections: " + super().__repr__() + "\n" + toString

    def _mutate_add_connection(self, mutation_record, node1, node2):
        """Only want linear graphs for data augmentation"""
        return True

    def mutate(self, mutation_record, attribute_magnitude=1, topological_magnitude=1, module_population=None, gen=-1):
        return super()._mutate(mutation_record, 0.1, 0, allow_connections_to_mutate=False, debug=False,
                               attribute_magnitude=attribute_magnitude, topological_magnitude=topological_magnitude)

    def inherit(self, genome):
        pass

    def to_phenotype(self, Phenotype=None):
        """Construct a data augmentation scheme from its genome"""

        da_scheme = AugmentationScheme(None, None)
        traversal = self.get_traversal_dictionary(exclude_disabled_connection=True)
        curr_node = self.get_input_node().id

        if not self._to_da_scheme(da_scheme, curr_node, traversal, debug=True):
            """all da's are disabled"""
            da_scheme.augs.append(AugmentationScheme.Augmentations["No_Operation"])

        gene_augs = []
        for node in self._nodes.values():
            if node.enabled():
                gene_augs.append(node.da())

        if len(gene_augs) != 0 and len(gene_augs) != len(da_scheme.augs):
            raise Exception(
                "failed to add all augmentations from gene. genes:" + repr(gene_augs) + "added:" + repr(da_scheme.augs))

        return da_scheme

    def _to_da_scheme(self, da_scheme: AugmentationScheme, curr_node_id, traversal_dictionary, debug=False):
        """auxillary method used to convert from da genome to phenotype"""
        this_node_added_da = False

        if self._nodes[curr_node_id].enabled():
            da_scheme.add_augmentation(self._nodes[curr_node_id].da)
            this_node_added_da = True

        if curr_node_id in traversal_dictionary:
            branches = 0

            for node_id in traversal_dictionary[curr_node_id]:
                branches += 1
                child_added_da = self._to_da_scheme(da_scheme, node_id, traversal_dictionary, debug=debug)

            if branches > 1:
                raise Exception("too many branches")

            return this_node_added_da or child_added_da

        return this_node_added_da

    def validate(self):
        """to be valid, a da individual must also be linear, ie: no branches"""
        return super().validate() and not self.has_branches()