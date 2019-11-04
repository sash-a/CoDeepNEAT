import copy
from typing import Tuple, Union, Type

from src2.Genotype.NEAT.Operators.Mutations.MutationRecord import MutationRecords
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome

from src2.Genotype.NEAT.Node import Node, NodeType
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome

input_node_params = (0, NodeType.INPUT)
output_node_params = (1, NodeType.OUTPUT)

hidden2_node = 2
hidden3_node = 3
hidden4_node = 4


def get_mini_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                    TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    """0 - 1"""
    mini_genome = TypeGenome([TypeNode(input_node_params), TypeNode(output_node_params)], [Connection(0, 0, 1)])
    return mini_genome, MutationRecords({(0, 1): 0}, 1, 0)


def get_small_linear_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                            TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    """0 -- 2 -- 1"""
    small_linear_genome = TypeGenome(
        [TypeNode(input_node_params), TypeNode(hidden2_node), TypeNode(output_node_params)],
        [Connection(1, 0, 2), Connection(2, 2, 1)])
    return small_linear_genome, MutationRecords({(0, 1): 0,  # initial mini genome
                                                 2: 0,  # add node on connection 0
                                                 (0, 2): 1,  # add connection for new node
                                                 (2, 1): 2},  # add connection for new node
                                                1, 0)


def get_small_tri_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                         TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    """
           0
         /  \
        1 -- 2
    """
    small_tri_genome = TypeGenome([TypeNode(input_node_params), TypeNode(hidden2_node), TypeNode(output_node_params)],
                                  [Connection(0, 0, 2), Connection(1, 0, 1), Connection(2, 2, 1)])

    return small_tri_genome, MutationRecords({(0, 1): 0,  # initial mini genome
                                              2: 0,  # add node on connection 0
                                              (0, 2): 1,  # add connection for new node
                                              (2, 1): 2},  # add connection for new node
                                             1, 0)
