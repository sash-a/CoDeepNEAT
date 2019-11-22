import random

from src2.Configuration import config
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from src2.Genotype.NEAT.Node import Node
from src2.Genotype.NEAT.Operators.Mutators.MutationReport import MutationReport
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT import MutationRecord
from src2.Genotype.NEAT.Operators.Mutators.GenomeMutator import GenomeMutator


class BlueprintGenomeMutator(GenomeMutator):

    def mutate(self, genome: Genome, mutation_record: MutationRecord):
        """
            performs base NEAT genome mutations, as well as node and genome property mutations
            as well as all mutations specific to blueprint genomes
        """
        mutation_report = self.mutate_base_genome(genome, mutation_record, add_node_chance=config.blueprint_add_node_chance,
                                add_connection_chance=config.blueprint_add_connection_chance)

        mutation_report += self.mutate_node_types(genome)
        mutation_report += self.mutate_species_numbers(genome)

        return genome

    def mutate_node_types(self, genome:Genome):
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

    def mutate_species_numbers(self, genome):
        mutation_report = MutationReport()
        import src2.main.Singleton as Singleton

        for node in genome.nodes.values():
            if type(node) != BlueprintNode:
                continue
            if random.random() < config.blueprint_node_species_switch_chance:
                possible_species_ids = [spc.id for spc in Singleton.instance.module_population.species]
                node.species_id: int = random.choice(possible_species_ids)
