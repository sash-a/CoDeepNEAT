import copy

from torch import nn

from src.Config import NeatProperties as Props
from src.DataAugmentation.AugmentationScheme import AugmentationScheme
from src.NEAT.Genome import Genome
from src.NEAT.Mutagen import Mutagen, ValueType
from src.Phenotype.BlueprintNode import BlueprintNode
from src.Phenotype.BlueprintGraph import BlueprintGraph
from src.Phenotype.ModuleNode import ModuleNode


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections, nodes)
        self.modules_used = []  # holds ref to module individuals used - can multiple represent
        self.modules_used_index = []  # hold tuple (species no, module index) of module used

        self.species_module_ref_map = {}  # maps species index: module ref in that species
        self.species_module_index_map = {}  # maps species index: module index in that species
        self.max_accuracy = 0  # the max accuracy of each of the samplings of this blueprint

        self.da_scheme: DAGenome = None
        # TODO make this a static number
        self.learning_rate = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.001, start_range=0.0006,
                                     end_range=0.003, print_when_mutating=False, mutation_chance=0.13)
        self.beta1 = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.9, start_range=0.88, end_range=0.92,
                             mutation_chance=0.1)
        self.beta2 = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.999, start_range=0.9988, end_range=0.9992,
                             mutation_chance=0.1)
        self.weight_init = Mutagen(nn.init.kaiming_uniform_, nn.init.xavier_uniform_,
                                   discreet_value=nn.init.kaiming_uniform_, name='initialization function',
                                   mutation_chance=0.13)
        self.da_scheme_index = -1

    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return BlueprintGraph(super().to_phenotype(BlueprintNode))

    def pick_da_scheme(self, da_population):
        if self.da_scheme is not None and self.da_scheme in da_population.species[0].members:
            self.da_scheme_index = da_population.species[0].members.index(self.da_scheme)
            # print("keeping existing DA scheme, taking new index:",self.da_scheme_index)
            return self.da_scheme

        # Assuming data augmentation only has 1 species
        # TODO make sure there is only ever 1 species - could make it random choice from individuals
        self.da_scheme, self.da_scheme_index = da_population.species[0].sample_individual(debug=False)
        # print("sampled new da scheme, index:",self.da_scheme_index)
        return self.da_scheme

    def inherit_species_module_mapping(self, other, acc):
        """Updates the species-module mapping if accuracy is higher than max accuracy"""
        if acc > self.max_accuracy:
            other.update_module_refs()
            self.species_module_ref_map = other.species_module_ref_map

            self.max_accuracy = acc

    def update_module_indexes(self, generation):
        self.species_module_index_map = {}
        for spc_index, module in self.species_module_ref_map.items():
            if module is None:
                print('Found a none node')
                continue

            if module in generation.module_population.species[spc_index]:
                self.species_module_index_map[spc_index] = generation.module_population.species[spc_index].index(module)

    def update_module_refs(self, generation):
        self.species_module_ref_map = {}
        for spc_index, module_index in self.species_module_index_map.items():
            self.species_module_ref_map[spc_index] = generation.module_population.species[spc_index][module_index]

    def mutate(self, mutation_record):
        return super()._mutate(mutation_record, Props.BP_NODE_MUTATION_CHANCE, Props.BP_CONN_MUTATION_CHANCE)

    def inherit(self, genome):
        self.da_scheme = genome.da_scheme
        self.weight_init = copy.deepcopy(genome.weight_init)
        # self.learning_rate = copy.deepcopy(genome.learning_rate)
        # self.beta1 = copy.deepcopy(genome.beta1)
        # self.beta2 = copy.deepcopy(genome.beta2)
        # print("inhereting from Blueprint genome an lr of:",self.learning_rate(), "and da sc:",self.da_scheme)

    def end_step(self, generation=None):
        super().end_step()
        self.modules_used = []
        self.modules_used_index = []
        self.max_accuracy = 0

        self.update_module_indexes(generation)
        # self.da_scheme_index = -1  # don't reset because bp holds onto its DA if it can

    def reset_number_of_module_species(self, num_module_species):
        for node in self._nodes.values():
            node.set_species_upper_bound(num_module_species)

    def get_all_mutagens(self):
        return [self.learning_rate, self.beta1, self.beta2, self.weight_init]


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

    def inherit(self, genome):
        pass

    def end_step(self, generation=None):
        super().end_step()
        self.module_node = None


class DAGenome(Genome):

    def __init__(self, connections, nodes):
        super().__init__(connections, nodes)

    def __repr__(self):
        node_names = []
        for node in self._nodes.values():
            node_names.append(node.get_node_name())

        toString = "\tNodes:" + repr(list(node_names)) + "\n" + "\tTraversal_Dict: " + repr(
            self._get_traversal_dictionary())
        return "\n" + "\tConnections: " + super().__repr__() + "\n" + toString

    def _mutate_add_connection(self, mutation_record, node1, node2):
        """Only want linear graphs for data augmentation"""
        return True

    def mutate(self, mutation_record):
        # print("mutating DA genome")
        return super()._mutate(mutation_record, 0.1, 0, allow_connections_to_mutate=False, debug=False)

    def inherit(self, genome):
        pass

    def to_phenotype(self, Phenotype=None):
        # Construct DA scheme from nodes
        # print("parsing",self, "to da scheme")
        da_scheme = AugmentationScheme(None, None)
        traversal = self._get_traversal_dictionary()
        curr_node = self.get_input_node().id

        if not self._to_da_scheme(da_scheme, curr_node, traversal):
            # self._to_da_scheme(da_scheme, curr_node, traversal,debug= True)
            """all da's are disabled"""
            da_scheme.augs.append(AugmentationScheme.Augmentations["No_Operation"])
            # raise Exception("never added any augmentations to pipeline. genome:", self)

        return da_scheme

    def _to_da_scheme(self, da_scheme: AugmentationScheme, curr_node_id, traversal_dictionary, debug=False):

        if curr_node_id not in traversal_dictionary:
            if debug:
                print("reached output node:", curr_node_id)
            return False

        added_an_aug = False
        for node_id in traversal_dictionary[curr_node_id]:
            if debug:
                print("visiting node:", node_id, "da_name:", self._nodes[node_id].da())
            da_name = self._nodes[node_id].da()
            # print("found da",da_name)
            if self._nodes[node_id].enabled():
                added_an_aug = True
                da_scheme.add_augmentation(self._nodes[node_id].da)
            added_an_aug = added_an_aug or self._to_da_scheme(da_scheme, node_id, traversal_dictionary)

        return added_an_aug

    def validate(self):
        return super().validate() and not self.has_branches()
