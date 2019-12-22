import random

from src2.configuration import config
from src2.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src2.genotype.cdn.nodes.module_node import ModuleNode
from src2.genotype.neat import mutation_record
from src2.genotype.neat.genome import Genome
from src2.genotype.neat.node import Node
from src2.genotype.neat.operators.mutators.genome_mutator import GenomeMutator
from src2.genotype.neat.operators.mutators.mutation_report import MutationReport


class BlueprintGenomeMutator(GenomeMutator):

    def mutate(self, genome: Genome, mutation_record: mutation_record):
        """
            performs base neat genome mutations, as well as node and genome property mutations
            as well as all mutations specific to blueprint genomes
        """
        mutation_report = self.mutate_base_genome(genome, mutation_record,
                                                  add_node_chance=config.blueprint_add_node_chance,
                                                  add_connection_chance=config.blueprint_add_connection_chance)

        mutation_report += self.mutate_node_types(genome)
        mutation_report += self.mutate_species_numbers(genome)
        mutation_report += self.forget_module_mappings_mutation(genome)

        return genome

    def mutate_node_types(self, genome:Genome) -> MutationReport:
        """
        chance to change nodes from blueprint nodes to module nodes and visa versa
        """
        mutation_report = MutationReport()

        if random.random() < config.blueprint_node_type_switch_chance:
            """chose 1 node to change type"""
            node: Node = random.choice(list(genome.nodes.values()))
            if type(node) == BlueprintNode:
                """change node to a module node"""
                module_node = ModuleNode(node.id,node.node_type)
                genome.nodes[module_node.id] = module_node
                mutation_report += "swapped blueprint node for a module node"

            if type(node) == ModuleNode:
                """change node back to a blueprint node"""
                blueprint_node = BlueprintNode(node.id,node.node_type)
                genome.nodes[blueprint_node.id] = blueprint_node
                mutation_report += "swapped module node for a blueprint node"

        return mutation_report

    def mutate_species_numbers(self, genome) -> MutationReport:
        mutation_report = MutationReport()
        import src2.main.singleton as Singleton

        for node in genome.nodes.values():
            if type(node) != BlueprintNode:
                continue
            if random.random() < config.blueprint_node_species_switch_chance:
                possible_species_ids = [spc.id for spc in Singleton.instance.module_population.species]
                new_species_id = random.choice(possible_species_ids)
                mutation_report+="changed species number of node " + str(node.id) + " from " + str(node.species_id) \
                                 + " to " + str(new_species_id)
                node.species_id = new_species_id

        return mutation_report

    def forget_module_mappings_mutation(self, genome) -> MutationReport:
        mutation_report = MutationReport()

        if config.use_module_retention and random.random()< config.module_map_forget_mutation_chance:
            choices = list(set([node.species_id for node in genome.nodes.values() if node.linked_module_id != -1]))
            if len(choices) == 0:
                return mutation_report

            species_id = random.choice(choices)
            for node in genome.nodes.values():
                if node.species_id == species_id:
                    node.linked_module_id = -1  # forget the link. will be sampled fresh next cycle

            mutation_report += "forgot module mapping for species " + str(species_id)

        return mutation_report

