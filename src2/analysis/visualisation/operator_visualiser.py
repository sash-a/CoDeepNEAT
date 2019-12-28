import copy
import random
from typing import List, Union

from runs import runs_manager
from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src2.genotype.cdn.mutators import blueprint_genome_mutator
from src2.genotype.cdn.mutators import module_genome_mutator
from src2.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src2.genotype.neat import mutation_record
from src2.genotype.neat.genome import Genome
from src2.genotype.neat.operators import cross
from src2.analysis.visualisation.genome_visualiser import get_graph_of
from test import static_genomes

"""
    class to visualise the various genome operations:
"""


def visualise_mutation(genomes: List[Genome],
                       mutator: Union[module_genome_mutator, blueprint_genome_mutator],
                       mutation_record: mutation_record, count=1, num_mutations=1):
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
        both_graph.view(directory=runs_manager.get_graphs_folder_path())


def visualise_crossover(genomes: List[Genome], count=1, parent_mutation_count = 1):
    """
    pairs genomes, plots parents along with their children
    """
    if len(genomes) < 2:
        raise Exception("need at least 2 genomes to show cross over, num found: " + str(len(genomes)))

    for i in range(count):
        parent1 = copy.deepcopy(random.choice(genomes))
        parent2 = None

        found_unique_parent = False
        while not found_unique_parent:
            parent2 = copy.deepcopy(random.choice(genomes))
            found_unique_parent = parent1 != parent2

        child = cross.over(parent1, parent2)
        if not child.validate():
            print("invalid child")
            continue

        parent1_graph = get_graph_of(parent1, node_names="parent1", sub_graph=True, label= "parent 1")
        parent2_graph = get_graph_of(parent2, node_names="parent2", sub_graph= True, label= "parent 2")
        full_graph = get_graph_of(child, node_names="full", node_colour= "yellow")
        full_graph.subgraph(parent1_graph)
        full_graph.subgraph(parent2_graph)

        full_graph.view(directory=runs_manager.get_graphs_folder_path())


if __name__ == "__main__":
    trivial_genome, record = static_genomes.get_mini_genome(BlueprintGenome, BlueprintNode)
    triangle_genome, record2 = static_genomes.get_small_tri_genome(BlueprintGenome, BlueprintNode)
    # three_chain_genome, record3 = StaticGenomes.get_small_linear_genome(BlueprintGenome, BlueprintNode)
    # large_genome, record4 = StaticGenomes.get_large_genome(BlueprintGenome, BlueprintNode)

    visualise_mutation([triangle_genome], blueprint_genome_mutator(), record2, num_mutations=200, count=1)
    # visualise_crossover([triangle_genome, trivial_genome], count=5)
