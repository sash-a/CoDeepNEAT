import copy

from src.Blueprint.Blueprint import BlueprintNode
from src.Config import NeatProperties as Props
from src.Module.ModuleNode import ModuleNode
from src.NEAT.Genome import Genome


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections, nodes)
        self.modules_used = []  # holds ref to module individuals used - can multiple represent

    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return super().to_phenotype(BlueprintNode)

    def mutate(self, mutation_record):
        return super()._mutate(mutation_record, Props.BP_NODE_MUTATION_CHANCE, Props.BP_CONN_MUTATION_CHANCE)

    def end_step(self):
        super().end_step()
        self.modules_used = []

    def reset_number_of_module_species(self, num_module_species):
        for node in self._nodes.values():
            node.set_species_upper_bound(num_module_species)

    def __repr__(self):
        return '------------------Blueprint structure------------------\n' + \
               super().__repr__() + \
               '\n------------------Modules used------------------\n' + \
               repr([repr(module) for module in self.modules_used])


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

    def to_phenotype(self, Phenotype):
        # Construct DA scheme from nodes
        pass
