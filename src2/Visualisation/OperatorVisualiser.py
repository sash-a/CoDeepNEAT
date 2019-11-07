import copy
import random
from typing import List, Union

from Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from Genotype.NEAT.Genome import Genome
from Genotype.NEAT.Operators import Cross
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
                       mutation_record: MutationRecord, count=1, num_mutations=1):
    """
    samples genomes and plots them before and after mutation
    """

    for i in range(count):
        genome: Genome = random.choice(genomes)
        genome = copy.deepcopy(genome)

        before_graph = get_graph_of(genome, node_names="before")

        mutant_genome = mutator.mutate(genome, mutation_record)
        for i in range(num_mutations - 1):
            # print("mutated genome")
            mutant_genome = mutator.mutate(mutant_genome, mutation_record)
        # print("mutated genome")

        both_graph = get_graph_of(mutant_genome, node_names="after", append_graph=before_graph, node_colour="yellow")
        both_graph.view()


def visualise_crossover(genomes: List[Genome], count=1):
    """
    pairs genomes, plots parents along with their children
    """
    if len(genomes) < 2:
        raise Exception("need at least 2 genomes to show cross over, num found: " + str(len(genomes)))

    for i in range(count):
        parent1 = random.choice(genomes)
        parent2 = None

        found_unique_parent = False
        while not found_unique_parent:
            parent2 = random.choice(genomes)
            found_unique_parent = parent1 != parent2

        child = Cross.over(parent1, parent2)
        parent1_graph = get_graph_of(parent1, node_names="parent1")
        parent2_graph = get_graph_of(parent2, node_names="parent2", append_graph=parent1_graph)
        full_graph = get_graph_of(child, node_names=" ")


if __name__ == "__main__":
    static_genome, record = StaticGenomes.get_mini_genome(BlueprintGenome, BlueprintNode)
    visualise_mutation([static_genome], BlueprintGenomeMutator(), record, num_mutations=500, count=5)
