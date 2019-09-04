import copy
import math
import random
from typing import List

from torch import nn

from src.CoDeepNEAT.CDNNodes import ModulenNEATNode, BlueprintNEATNode
from src.Config import Config, NeatProperties as Props
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
        self.modules_used_index = []  # hold tuple (species no, module index) of module used

        self.species_module_ref_map = {}  # maps species index: module ref in that species
        self.species_module_index_map = {}  # maps species index: module index in that species
        self.max_accuracy = 0  # the max accuracy of each of the samplings of this blueprint

        self.da_scheme: DAGenome = None
        self.da_scheme_index = -1

        # TODO make this a static number
        self.learning_rate = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.001, start_range=0.0006,
                                     end_range=0.003, print_when_mutating=False, mutation_chance=0)
        self.beta1 = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.9, start_range=0.88, end_range=0.92,
                             mutation_chance=0)
        self.beta2 = Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.999, start_range=0.9988, end_range=0.9992,
                             mutation_chance=0)
        self.weight_init = Mutagen(nn.init.kaiming_uniform_, nn.init.xavier_uniform_,
                                   discreet_value=nn.init.kaiming_uniform_, name='initialization function',
                                   mutation_chance=0.13)

    if Config.use_representative:
        representatives: List[ModulenNEATNode] = property(lambda self: self.get_all_reps())

    def get_all_reps(self) -> List[ModulenNEATNode]:
        if not Config.use_representative:
            raise Exception('Use representatives is false, but get all representatives was called')

        reps = list()
        for node in self._nodes.values():
            if not isinstance(node, BlueprintNEATNode):
                raise Exception('Type: ' + str(type(node) + ' stored as blueprint node'))

            reps.append(node.representative)

        return reps

    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return BlueprintGraph(super().to_phenotype(BlueprintNode))

    def pick_da_scheme(self, da_population):
        if self.da_scheme is not None and self.da_scheme in da_population.species[0].members:
            self.da_scheme_index = da_population.species[0].members.index(self.da_scheme)
            # print("keeping existing DA scheme, taking new index:", self.da_scheme_index)
            return self.da_scheme

        # Assuming data augmentation only has 1 species
        # TODO make sure there is only ever 1 species - could make it random choice from individuals
        self.da_scheme, self.da_scheme_index = da_population.species[0].sample_individual()
        # print("sampled new da scheme, index:",self.da_scheme_index)
        return self.da_scheme

    def inherit_species_module_mapping(self, generation, other, acc, da_scheme=None, inherit_module_mapping=True):
        """Updates the species-module mapping if accuracy is higher than max accuracy"""
        if acc > self.max_accuracy:
            if inherit_module_mapping:
                other.update_module_refs(generation)
                self.species_module_ref_map = other.species_module_ref_map
                # print('inherited mapping:', self.species_module_ref_map)

            self.max_accuracy = acc

            if da_scheme is not None:
                self.da_scheme = da_scheme

    def update_module_indexes(self, generation):
        self.species_module_index_map = {}

        if Config.use_representative:
            for rep, module in self.species_module_ref_map.items():
                if module is None:
                    continue

                for species_index, species in enumerate(generation.module_population.species):
                    if module in species:
                        self.species_module_index_map[rep] = \
                            (species_index, generation.module_population.species[species_index].members.index(module))
                        break
        else:
            for spc_index, module in self.species_module_ref_map.items():
                if module is None:
                    continue

                if spc_index < len(generation.module_population.species) and \
                        module in generation.module_population.species[spc_index]:

                    self.species_module_index_map[spc_index] = \
                        generation.module_population.species[spc_index].members.index(module)
                    # print("Found a live module")

                elif Config.allow_cross_species_mappings:
                    for new_species_index, species in enumerate(generation.module_population.species):
                        if module in species:
                            """found module in new species"""
                            # print("making overide mapping from",spc_index,"to",new_species_index,generation.module_population.species[new_species_index].members.index(module))
                            self.species_module_index_map[spc_index] = \
                                (new_species_index,
                                 generation.module_population.species[new_species_index].members.index(module))
                            break

    def update_module_refs(self, generation):
        self.species_module_ref_map = {}

        if Config.use_representative:
            reps = self.representatives
            for rep, (spc_index, module_index) in self.species_module_index_map.items():
                if rep not in reps:  # removes reps that no longer exist
                    continue
                self.species_module_ref_map[rep] = generation.module_population.species[spc_index][module_index]
        else:
            for spc_index, module_index in self.species_module_index_map.items():
                if isinstance(module_index, tuple):
                    """this is an override index. this module is found in a different species"""
                    if not Config.allow_cross_species_mappings:
                        raise Exception('Cross species mapping disabled, but received tuple as value in map')
                    spc, mod = module_index
                    self.species_module_ref_map[spc_index] = generation.module_population.species[spc][mod]
                    # print("using override mapping from:",spc_index,"to",spc,mod,"to update refs")
                else:
                    self.species_module_ref_map[spc_index] = generation.module_population.species[spc_index][
                        module_index]

    def mutate(self, mutation_record, attribute_magnitude=1, topological_magnitude=1, module_population=None, gen=-1):
        if Config.module_retention and random.random() < 0.1 * topological_magnitude and self.species_module_ref_map:
            # release a module_individual
            tries = 100

            while tries > 0:
                species_no = random.choice(list(self.species_module_ref_map.keys()))
                if self.species_module_ref_map[species_no] is not None:
                    self.species_module_ref_map[species_no] = None
                    break
                tries -= 1

        if Config.evolve_data_augmentations and random.random() < 0.2:
            self.da_scheme = None

        if Config.use_representative:
            reps = self.representatives
            for node in self._nodes.values():
                if gen == -1:
                    raise Exception('Invalid generation number: -1')

                chance = Config.rep_mutation_chance_early if gen <= 3 else Config.rep_mutation_chance_late
                if random.random() > chance:  # no rep mutation
                    continue

                old_rep = copy.deepcopy(node.representative)
                new_rep = node.choose_representative(module_population.individuals, reps)

                # Chance to mutate all nodes with the same representative
                if random.random() < Config.similar_rep_mutation_chance:
                    for other_node in self._nodes.values():
                        if other_node.representative == old_rep:
                            other_node.representative = new_rep

        nodes_before_mutation = set(self._nodes.keys())
        mutated = super()._mutate(mutation_record, Props.BP_NODE_MUTATION_CHANCE, Props.BP_CONN_MUTATION_CHANCE,
                                  attribute_magnitude=attribute_magnitude, topological_magnitude=topological_magnitude)
        # Check if a node was added
        if Config.use_representative:
            for node_id in self._nodes.keys():
                if node_id not in nodes_before_mutation:
                    self._nodes[node_id].choose_representative(module_population.individuals, reps)

        return mutated

    def inherit(self, genome):
        self.da_scheme = genome.da_scheme
        self.weight_init = copy.deepcopy(genome.weight_init)
        self.species_module_ref_map = genome.species_module_ref_map

    def end_step(self, generation=None):
        super().end_step()
        self.modules_used = []
        self.modules_used_index = []
        self.max_accuracy = 0

        self.update_module_indexes(generation)

    def reset_number_of_module_species(self, num_module_species, generation_number):
        for node in self._nodes.values():
            node.set_species_upper_bound(num_module_species, generation_number)

    def get_all_mutagens(self):
        return [self.learning_rate, self.beta1, self.beta2, self.weight_init]


class ModuleGenome(Genome):
    def __init__(self, connections, nodes):
        super(ModuleGenome, self).__init__(connections, nodes)
        self.module_node = None  # the module node created from this gene

    def __eq__(self, other):
        if not isinstance(other, ModuleGenome):
            raise TypeError(str(type(other)) + ' cannot be equal to ModuleGenome')

        self_ids = [nodeid for nodeid in self._nodes.keys()] + [connid for connid in self._connections.keys()]
        other_ids = [nodeid for nodeid in other._nodes.keys()] + [connid for connid in other._connections.keys()]

        self_attribs = []
        for node in self._nodes.values():
            for mutagen in node.get_all_mutagens():
                self_attribs.extend(mutagen.get_all_sub_values())

        other_attribs = []
        for node in other._nodes.values():
            for mutagen in node.get_all_mutagens():
                other_attribs.extend(mutagen.get_all_sub_values())

        return self_ids == other_ids and self_attribs == other_attribs

    def __hash__(self):
        self_ids = [nodeid for nodeid in self._nodes.keys()] + [connid for connid in self._connections.keys()]
        attribs = []
        for node in self._nodes.values():
            for mutagen in node.get_all_mutagens():
                attribs.extend(mutagen.get_all_sub_values())

        return hash(tuple(self_ids + attribs))

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

    def distance_to(self, other):
        if type(self) != type(other):
            raise TypeError('Trying finding distance from Module genome to ' + str(type(other)))

        attrib_dist = self.get_attribute_distance(other)
        topology_dist = self.get_topological_distance(other)

        # print(attrib_dist, topology_dist, math.sqrt(attrib_dist * attrib_dist + topology_dist * topology_dist))
        return math.sqrt(attrib_dist * attrib_dist + topology_dist * topology_dist)

    def get_attribute_distance(self, other):
        if not isinstance(other, ModuleGenome):
            raise TypeError('Expected type of ModuleGenome, received type: ' + str(type(other)))

        attrib_dist = 0
        common_nodes = self._nodes.keys() & other._nodes.keys()

        for node_id in common_nodes:
            self_node, other_node = self._nodes[node_id], other._nodes[node_id]
            for self_mutagen, other_mutagen in zip(self_node.get_all_mutagens(), other_node.get_all_mutagens()):
                attrib_dist += self_mutagen.distance_to(other_mutagen)

        attrib_dist /= len(common_nodes)
        return attrib_dist

    def mutate(self, mutation_record, attribute_magnitude=1, topological_magnitude=1, module_population=None, gen=-1):
        return super()._mutate(mutation_record, Props.MODULE_NODE_MUTATION_CHANCE, Props.MODULE_CONN_MUTATION_CHANCE,
                               attribute_magnitude=attribute_magnitude, topological_magnitude=topological_magnitude)

    def inherit(self, genome):
        pass

    def end_step(self, generation=None):
        super().end_step()
        self.module_node = None

    # def __repr__(self):
    #     return '\n------------------Connections--------------\n' + repr(self._connections) + \
    #            '\n---------------------Nodes-----------------\n' + repr(self._nodes)
    def __repr__(self):
        return str(hash(self))

    def get_comlexity(self):
        complexity = 0
        for node in self._nodes.values():
            complexity += node.get_complexity()
        return complexity


class DAGenome(Genome):
    def __init__(self, connections, nodes):
        super().__init__(connections, nodes)

    def __repr__(self):
        node_names = []
        for node in self._nodes.values():
            if node.enabled():
                node_names.append(node.get_node_name())

        toString = "\tNodes:" + repr(list(node_names)) + "\n" + "\tTraversal_Dict: " + repr(
            self._get_traversal_dictionary())
        return "\n" + "\tConnections: " + super().__repr__() + "\n" + toString

    def _mutate_add_connection(self, mutation_record, node1, node2):
        """Only want linear graphs for data augmentation"""
        return True

    def mutate(self, mutation_record, attribute_magnitude=1, topological_magnitude=1, module_population=None, gen=-1):
        # print("mutating DA genome")
        return super()._mutate(mutation_record, 0.1, 0, allow_connections_to_mutate=False, debug=False,
                               attribute_magnitude=attribute_magnitude, topological_magnitude=topological_magnitude)

    def inherit(self, genome):
        pass

    def to_phenotype(self, Phenotype=None):
        # Construct DA scheme from nodes
        # print("parsing",self, "to da scheme")
        da_scheme = AugmentationScheme(None, None)
        traversal = self._get_traversal_dictionary(exclude_disabled_connection=True)
        curr_node = self.get_input_node().id

        if not self._to_da_scheme(da_scheme, curr_node, traversal, debug=True):
            # self._to_da_scheme(da_scheme, curr_node, traversal,debug= True)
            """all da's are disabled"""
            # print("added no da's from gene. adding in NOOP")
            da_scheme.augs.append(AugmentationScheme.Augmentations["No_Operation"])
            # raise Exception("never added any augmentations to pipeline. genome:", self)

        gene_augs = []
        for node in self._nodes.values():
            if node.enabled():
                gene_augs.append(node.da())

        if len(gene_augs) != 0 and len(gene_augs) != len(da_scheme.augs):
            raise Exception(
                "failed to add all augs from gene. genes:" + repr(gene_augs) + "added:" + repr(da_scheme.augs))

        return da_scheme

    def _to_da_scheme(self, da_scheme: AugmentationScheme, curr_node_id, traversal_dictionary, debug=False):

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
        return super().validate() and not self.has_branches()
