from configuration import config
from src.genotype.neat import mutation_record
from src.genotype.neat.genome import Genome
from src.genotype.neat.operators.mutators.genome_mutator import GenomeMutator


class ModuleGenomeMutator(GenomeMutator):

    def mutate(self, genome: Genome, mutation_record: mutation_record):
        """
            performs base neat genome mutations, as well as node and genome property mutations
            as well as all mutations specific to module genomes
        """
        mutation_report = self.mutate_base_genome(genome, mutation_record, add_node_chance=config.blueprint_add_node_chance,
                                add_connection_chance=config.blueprint_add_connection_chance)

        # genome.visualize()
        # print(mutation_report)
