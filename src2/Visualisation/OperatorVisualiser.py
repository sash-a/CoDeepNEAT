import random
from typing import List, Union, Type

from Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from Genotype.NEAT.Genome import Genome
from Genotype.NEAT.Operators.Mutations import MutationRecord
from Genotype.NEAT.Operators.Mutations.BlueprintGenomeMutator import BlueprintGenomeMutator
from Genotype.NEAT.Operators.Mutations.GenomeMutator import GenomeMutator
from Genotype.NEAT.Operators.Mutations.ModuleGenomeMutator import ModuleGenomeMutator
from Visualisation.GenomeVisualiser import get_graph_of
from test import StaticGenomes

"""
    class to visualise the various genome operations:
"""


def visualise_mutation(genomes: List[Genome],
                       mutator: Union[GenomeMutator, ModuleGenomeMutator, BlueprintGenomeMutator],
                       mutation_record: MutationRecord, count=1, num_mutations = 1):
    """
    samples genomes and plots them before and after mutation
    """

    for i in range(count):
        genome: Genome = random.choice(genomes)
        before_graph = get_graph_of(genome, node_names="before")

        mutant_genome = mutator.mutate(genome, mutation_record)
        for i in range(num_mutations -1):
            mutant_genome = mutator.mutate(mutant_genome, mutation_record)
        both_graph = get_graph_of(mutant_genome, node_names="after", append_graph=before_graph)
        both_graph.view()


def visualise_crossover(genomes: List[Genome], count=1):
    """
    pairs genomes, plots parents along with their children
    """
    pass


if __name__ == "__main__":
    static_genome, record = StaticGenomes.get_small_tri_genome(BlueprintGenome, BlueprintNode)
    visualise_mutation([static_genome], BlueprintGenomeMutator(), record, num_mutations= 2)
