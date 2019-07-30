import copy
import random

from torch import nn

from src.Config import NeatProperties as Props
from src.DataAugmentation.AugmentationScheme import AugmentationScheme
from src.NEAT.Genome import Genome
from src.NEAT.Mutagen import Mutagen, ValueType
from src.Phenotype.BlueprintGraph import BlueprintGraph
from src.Phenotype.BlueprintNode import BlueprintNode
from src.Phenotype.ModuleNode import ModuleNode


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections, nodes)
        self.modules_used = []  # holds ref to module individuals used - can multiple represent
        self.species_module_mapping = {}
        self.best_evaluation_accuracy = 0
        self.modules_used_index = []  # hold tuple (species no, module index) of module used
        self.da_scheme: DAGenome = None
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

    def to_blueprint_graph(self):
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

    def mutate(self, mutation_record):
        if random.random() < 0.1 and len(self.species_module_mapping) > 0:
            """release a module_individual"""
            tries = 100

            while tries > 0:
                species_no = random.choice(list(self.species_module_mapping.keys()))
                if self.species_module_mapping[species_no] is not None:
                    self.species_module_mapping[species_no] = None
                    break
                tries-=1

        return super()._mutate(mutation_record, Props.BP_NODE_MUTATION_CHANCE, Props.BP_CONN_MUTATION_CHANCE)

    def inherit(self, genome):
        self.da_scheme = genome.da_scheme
        self.learning_rate = copy.deepcopy(genome.learning_rate)
        self.beta1 = copy.deepcopy(genome.beta1)
        self.beta2 = copy.deepcopy(genome.beta2)
        # print("inhereting from Blueprint genome an lr of:",self.learning_rate(), "and da sc:",self.da_scheme)
        self.species_module_mapping = genome.species_module_mapping  # TODO look up deep/ shallow
        print("inheriting module mapping:",self.species_module_mapping)

    def inherit_species_module_mapping_from_phenotype(self, species_module_mapping, accuracy):
        if accuracy > self.best_evaluation_accuracy:
            """
            a blueprint individual can be evaluated multiple times usiing multiple blueprint graphs
            each will have a common set of modules they have a hold on, as well as a set of nodes which do not have a module
            each blueprint graph will pick different modules for their unalloted nodes
            the blueprint graph which performs best passes its species:module mapping to the genome to be used in the future by self/children
            """

            if self.best_evaluation_accuracy >  0:
                print(self,"updating species mapping with:", self.species_module_mapping)
            self.best_evaluation_accuracy = accuracy
            self.species_module_mapping = species_module_mapping


    def end_step(self):
        super().end_step()
        self.modules_used = []
        self.modules_used_index = []
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
            raise Exception("never added any augmentations to pipeline. genome:", self)

        return da_scheme

    def _to_da_scheme(self, da_scheme: AugmentationScheme, curr_node_id, traversal_dictionary):
        if curr_node_id not in traversal_dictionary:
            return False

        added_an_aug = False
        for node_id in traversal_dictionary[curr_node_id]:
            da_name = self._nodes[node_id].da()
            # print("found da",da_name)
            if self._nodes[node_id].enabled():
                added_an_aug = True
                da_scheme.add_augmentation(self._nodes[node_id].da)
            added_an_aug = added_an_aug or self._to_da_scheme(da_scheme, node_id, traversal_dictionary)

        return added_an_aug

    def validate(self):
        return super().validate() and not self.has_branches()
