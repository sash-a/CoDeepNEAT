from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT import MutationRecord
from src2.Genotype.NEAT.Operators.Mutators.GenomeMutator import GenomeMutator
from src2.Configuration import config


class DataAugmentationGenomeMutator(GenomeMutator):

    def mutate(self, genome: Genome, mutation_record: MutationRecord):
        """
            performs base NEAT genome mutations, as well as node and genome property mutations
            as well as all mutations specific to data augmentation genomes
        """
        self.mutate_base_genome(genome, mutation_record, config.add_da_node_chance, 0)  # todo get these mutation chances from config
