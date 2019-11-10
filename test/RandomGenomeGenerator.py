import random
from typing import Tuple, Union, Type

from Genotype.NEAT.MutationRecord import MutationRecords
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome

from src2.Genotype.NEAT.Node import Node, NodeType
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome

input_node_params = (0, NodeType.INPUT)
output_node_params = (1, NodeType.OUTPUT)


def random_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                  TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    genome = TypeGenome(
        [TypeNode(input_node_params), TypeNode(2), TypeNode(3), TypeNode(4), TypeNode(5), TypeNode(output_node_params)],
        [Connection(1, 0, 2), Connection(3, 2, 3), Connection(7, 3, 4), Connection(8, 4, 5), Connection(9, 5, 1)])
    mr = MutationRecords({(0, 1): 0,  # initial mini genome
                          0: 2,  # add node 2 on connection 0
                          (0, 2): 1,  # add connection for new node
                          (2, 1): 2,  # add connection for new node
                          2: 3,
                          (2, 3): 3,
                          (3, 1): 4,
                          4: 4,
                          (3, 4): 7,
                          (4, 1): 10,
                          10: 5,
                          (4, 5): 8,
                          (5, 1): 9},
                         5, 10)

    for _ in random.randint(1, 10):  # random number of mutations
        if random.choice([True, False]):
            # Add connection
            from_id = random.randint(0, 6)
            to_id = random.randint(0, 6)
            if mr.exists((from_id, to_id)):
                new_id = mr.connection_mutations[(from_id, to_id)]
            else:
                new_id = mr.add_mutation((from_id, to_id))
            genome.add_connection(Connection(new_id, from_id, to_id))
        else:
            # Add node
            # TODO once mutation records have been fixed
            pass

    return genome, mr
