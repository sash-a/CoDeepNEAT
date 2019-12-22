from src2.configuration import config
from src2.genotype.neat import mutation_record
from src2.genotype.neat.genome import Genome
from src2.genotype.neat.operators.mutators.genome_mutator import GenomeMutator


class DataAugmentationGenomeMutator(GenomeMutator):

    def mutate(self, genome: Genome, mutation_record: mutation_record):
        """
            performs base neat genome mutations, as well as node and genome property mutations
            as well as all mutations specific to data augmentation genomes
        """
        self.mutate_base_genome(genome, mutation_record, config.add_da_node_chance, 0)  # todo get these mutation chances from config
