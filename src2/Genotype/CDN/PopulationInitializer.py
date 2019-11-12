from __future__ import annotations

from typing import Union, Type, TYPE_CHECKING, List

from src2.Configuration import config
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.MutationRecord import MutationRecords
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode, NodeType

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
    from src2.Genotype.CDN.Genomes.DAGenome import DAGenome

    from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
    from src2.Genotype.CDN.Nodes.DANode import DANode


def create_population(pop_size: int, Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
                      Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        List[Union[ModuleGenome, BlueprintGenome, DAGenome]]:
    pop = []
    for _ in range(pop_size):
        pop.append(_create_individual(Node, Genome))

    return pop


def _create_individual(Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
                       Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        Union[ModuleGenome, BlueprintGenome, DAGenome]:
    in_node = Node(0, NodeType.INPUT)
    out_node = Node(1, NodeType.OUTPUT)

    # Making the in and out nodes of modules blank
    if Node == ModuleNode and config.blank_io_nodes:
        _blank_node(in_node)
        _blank_node(out_node)

    return Genome(
        [in_node, Node(2, NodeType.HIDDEN), out_node],
        [Connection(0, 0, 2), Connection(1, 2, 1)]  # TODO should this have the connection from 0 -> 1?
    )


def create_mr() -> MutationRecords:
    return MutationRecords({(0, 2): 0, (2, 1): 1}, {}, 2, 2)  # TODO should we add in the connection from 0 -> 1?


def _blank_node(node: ModuleNode):
    """Makes a module node only return its input and doesn't allow it to change"""
    node.layer_type.set_value(None)

    dropout = node.layer_type.get_submutagen('dropout')
    regularisation = node.layer_type.get_submutagen('regularisation')
    dropout.set_value(None)
    regularisation.set_value(None)
    node.layer_type.mutation_chance = 0
    dropout.mutation_chance = 0
    regularisation.mutation_chance = 0
