import copy

from src.Blueprint.Blueprint import BlueprintNode
from src.Config import NeatProperties as Props
from src.Module.ModuleNode import ModuleNode
from src.NEAT.Genome import Genome
from src.NEAT.Mutagen import Mutagen,ValueType


from src.DataAugmentation.AugmentationScheme import AugmentationScheme


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections, nodes)
        self.modules_used = []  # holds ref to module individuals used - can multiple represent
        self.da_scheme: DAGenome = None
        self.learning_rate = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.001, start_range= 0.0003, end_range= 0.005, print_when_mutating=False)
        self.beta1 = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.9, start_range= 0.87, end_range= 0.93)
        self.beta2 = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.999, start_range= 0.9987, end_range= 0.9993)


    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return super().to_phenotype(BlueprintNode)

    def pick_da_scheme(self, da_population):
        if self.da_scheme is not None and self.da_scheme in da_population.species[0]:
            return self.da_scheme

        # Assuming data augmentation only has 1 species
        self.da_scheme, _ = da_population.species[0].sample_individual()
        return self.da_scheme

    def mutate(self, mutation_record):
        return super()._mutate(mutation_record, Props.BP_NODE_MUTATION_CHANCE, Props.BP_CONN_MUTATION_CHANCE)

    def end_step(self):
        super().end_step()
        self.modules_used = []

    def reset_number_of_module_species(self, num_module_species):
        for node in self._nodes.values():
            node.set_species_upper_bound(num_module_species)

    def get_all_mutagens(self):
        return [self.learning_rate, self.beta1, self.beta2]


class ModuleGenome(Genome):

    def __init__(self, connections, nodes):
        super(ModuleGenome, self).__init__(connections, nodes)
        self.module_node = None  # the module node created from this gene

    def to_module(self):
        """
        returns the stored module_node of this gene, or generates and returns it if module_node is null
        :return: the module graph this individual represents
        """
        if self.module_node is not None:
            # print("module genome already has module - returning a copy")
            return copy.deepcopy(self.module_node)

        module = super().to_phenotype(ModuleNode)
        self.module_node = module
        return copy.deepcopy(module)

    def mutate(self, mutation_record):
        return super()._mutate(mutation_record, Props.MODULE_NODE_MUTATION_CHANCE, Props.MODULE_CONN_MUTATION_CHANCE)

    def end_step(self):
        super().end_step()
        self.module_node = None


class DAGenome(Genome):
    def __init__(self, connections, nodes):
        super().__init__(connections, nodes)

    def _mutate_add_connection(self, mutation_record, node1, node2):
        """Only want linear graphs for data augmentation"""
        return True

    def mutate(self, mutation_record):
        return super()._mutate(mutation_record, 0.1, 0, allow_connections_to_mutate=False)

    def to_phenotype(self, Phenotype=None):
        # Construct DA scheme from nodes
        #print("parsing",self, "to da scheme")
        da_scheme = AugmentationScheme(None, None)
        traversal = self._get_traversal_dictionary()
        curr_node = self.get_input_node().id

        self._to_da_scheme(da_scheme, curr_node, traversal)

        return da_scheme

    def _to_da_scheme(self, da_scheme: AugmentationScheme, curr_node_id, traversal_dictionary):
        if curr_node_id not in traversal_dictionary:
            return

        for node_id in traversal_dictionary[curr_node_id]:
            da_name = self._nodes[node_id].da()
            #print("found da",da_name)
            if self._nodes[node_id].enabled():
                da_scheme.add_augmentation(self._nodes[node_id].da)
            self._to_da_scheme(da_scheme, node_id, traversal_dictionary)
