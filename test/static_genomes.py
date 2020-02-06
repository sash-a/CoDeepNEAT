from __future__ import annotations

from typing import Tuple, Union, Type, TYPE_CHECKING

from src.genotype.neat.connection import Connection
from src.genotype.neat.genome import Genome
from src.genotype.neat.mutation_record import MutationRecords
from src.genotype.neat.node import Node, NodeType

if TYPE_CHECKING:
    from src.genotype.cdn.nodes.module_node import ModuleNode
    from src.genotype.cdn.nodes.blueprint_node import BlueprintNode
    from src.genotype.cdn.genomes.module_genome import ModuleGenome
    from src.genotype.cdn.genomes.BlueprintGenome import BlueprintGenome

input_node_params = (0, NodeType.INPUT)
output_node_params = (1, NodeType.OUTPUT)

hidden2_node = 2
hidden3_node = 3
hidden4_node = 4


def get_mini_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                    TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    """0 - 1"""
    mini_genome = TypeGenome([TypeNode(*input_node_params), TypeNode(*output_node_params)], [Connection(0, 0, 1)])
    return mini_genome, MutationRecords({(0, 1): 0}, {}, 1, 0)


def get_small_linear_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                            TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    """0 -- 2 -- 1"""
    small_linear_genome = TypeGenome(
        [TypeNode(*input_node_params), TypeNode(hidden2_node, NodeType.HIDDEN), TypeNode(*output_node_params)],
        [Connection(1, 0, 2), Connection(2, 2, 1)])
    return small_linear_genome, MutationRecords({(0, 1): 0,  # initial mini genome
                                                 0: 2,  # add node on connection 0
                                                 (0, 2): 1,  # add connection for new node
                                                 (2, 1): 2},  # add connection for new node
                                                2, 2)


def get_small_tri_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                         TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    """
           0
         /  \
        1 -- 2
    """
    small_tri_genome = TypeGenome(
        [TypeNode(*input_node_params), TypeNode(hidden2_node, NodeType.HIDDEN), TypeNode(*output_node_params)],
        [Connection(0, 0, 1), Connection(1, 0, 2), Connection(2, 2, 1)])
    return small_tri_genome, MutationRecords({(0, 1): 0,  # initial mini genome
                                              # add node on connection 0
                                              (0, 2): 1,  # add connection for new node
                                              (2, 1): 2},  # add connection for new node
                                             {(0, 0): 2},
                                             2, 2)


def get_large_genome(TypeGenome: Union[Type[Genome], Type[ModuleGenome], Type[BlueprintGenome]],
                     TypeNode: Union[Type[Node], Type[ModuleNode], Type[BlueprintNode]]) \
        -> Tuple[Union[Genome, ModuleGenome, BlueprintGenome], MutationRecords]:
    """
            1
           / \
          4   3
          |   /
          \  2
           \/
           0
    """
    large_genome = TypeGenome(
        [TypeNode(*input_node_params), TypeNode(hidden2_node, NodeType.HIDDEN), TypeNode(hidden3_node, NodeType.HIDDEN),
         TypeNode(hidden4_node, NodeType.HIDDEN), TypeNode(*output_node_params)],
        [Connection(1, 0, 2), Connection(3, 2, 3), Connection(4, 3, 1), Connection(5, 0, 4), Connection(6, 4, 1),
         Connection(6, 1, 1)])

    return large_genome, MutationRecords({(0, 1): 0,
                                          0: 2,  # add node on connection 0
                                          (0, 2): 1,  # add connection for new node
                                          (2, 1): 2,  # add connection for new node
                                          2: 3,  # Add node id=3 on connection id=2
                                          (2, 3): 3,  # Add connection for node 3
                                          (3, 1): 4,  # Add connection for node 3
                                          # TODO: this is the problem with mutation records. A lookup would've been done
                                          #  for an add node on connection 0 and this node would've been given id=2, but
                                          #  that is already in the genome elsewhere...
                                          0: 4,  # Add node id=4 on connection id=0
                                          (0, 4): 5,  # Add connection for node 4
                                          (4, 1): 6},  # Add connection for node 4
                                         3, 6)


if __name__ == "__main__":
    import importlib.util

    # print(get_large_genome(Genome, Node)[0].has_cycle())\
    gen = get_small_tri_genome(Genome, Node)[0]
    print(gen.get_traversal_dictionary(exclude_disabled_connection=True))
    print(gen.get_reachable_nodes(False))

    spec = importlib.util.spec_from_file_location("evolved_augmentations.py",
                                                  "/home/sasha/Documents/CoDeepNEAT/src/phenotype/augmentations/evolved_augmentations.py")
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    print(foo.augmentations)
