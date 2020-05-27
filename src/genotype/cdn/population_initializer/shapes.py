from __future__ import annotations

from typing import Union, Type, TYPE_CHECKING

from src.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src.genotype.cdn.nodes.module_node import ModuleNode, NodeType
from src.genotype.neat.connection import Connection
from src.genotype.neat.mutation_record import MutationRecords

if TYPE_CHECKING:
    from src.genotype.cdn.genomes.module_genome import ModuleGenome
    from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
    from src.genotype.cdn.genomes.da_genome import DAGenome
    from src.genotype.cdn.nodes.da_node import DANode

in_node_params = (0, NodeType.INPUT)
out_node_params = (1, NodeType.OUTPUT)
mid_node_params = (2, NodeType.HIDDEN)


# Creates genomes with the named shapes, of the given type
def io_only(Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
            Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        Union[ModuleGenome, BlueprintGenome, DAGenome]:
    return Genome(
        ([Node(*in_node_params), Node(*out_node_params)]),
        [Connection(1, 0, 1)]
    )


def linear(Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
           Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        Union[ModuleGenome, BlueprintGenome, DAGenome]:
    return Genome(
        ([Node(*in_node_params), Node(*mid_node_params), Node(*out_node_params)]),
        [Connection(1, 0, 2), Connection(2, 2, 1)]
    )


def triangle(Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
             Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        Union[ModuleGenome, BlueprintGenome, DAGenome]:
    return Genome(
        ([Node(*in_node_params), Node(*mid_node_params), Node(*out_node_params)]),
        [Connection(0, 0, 1), Connection(1, 0, 2), Connection(2, 2, 1)]
    )


def diamond(Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
            Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        Union[ModuleGenome, BlueprintGenome, DAGenome]:
    return Genome(
        ([Node(*in_node_params), Node(*mid_node_params), Node(3, NodeType.HIDDEN), Node(*out_node_params)]),
        [Connection(1, 0, 2), Connection(3, 0, 3), Connection(4, 3, 1), Connection(2, 2, 1)]
    )


str_to_shape = {
    'io_only': io_only,
    'linear': linear,
    'triangle': triangle,
    'diamond': diamond
}


# Creates the mutation record for the given shapes
def io_only_mr(mr: MutationRecords):
    mutation = (0, 1)
    if not mr.exists(mutation, True):
        mr.add_mutation(mutation, True)


def linear_mr(mr: MutationRecords):
    io_only_mr(mr)

    node_mutation = (0, 0)
    conn_mutations = [(0, 2), (2, 1)]
    if not mr.exists(node_mutation, False):
        mr.add_mutation(node_mutation, False)

    for conn_mutation in conn_mutations:
        if not mr.exists(conn_mutation, True):
            mr.add_mutation(conn_mutation, True)


def triangle_mr(mr: MutationRecords):
    linear_mr(mr)


def diamond_mr(mr: MutationRecords):
    linear_mr(mr)

    node_mutation = (0, 1)
    conn_mutations = [(0, 3), (3, 1)]
    if not mr.exists(node_mutation, False):
        mr.add_mutation(node_mutation, False)

    for conn_mutation in conn_mutations:
        if not mr.exists(conn_mutation, True):
            mr.add_mutation(conn_mutation, True)


str_to_shape_mr = {
    'io_only': io_only_mr,
    'linear': linear_mr,
    'triangle': triangle_mr,
    'diamond': diamond_mr
}
