from __future__ import annotations

from typing import Union, Type, TYPE_CHECKING, List

from src2.Configuration import config
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.MutationRecord import MutationRecords
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode, NodeType
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
    from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
    from src2.Genotype.CDN.Genomes.DAGenome import DAGenome

    from src2.Genotype.CDN.Nodes.DANode import DANode


def create_population(pop_size: int, Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
                      Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        List[Union[ModuleGenome, BlueprintGenome, DAGenome]]:
    pop = []
    for _ in range(pop_size // 3 + 1):
        pop.extend(_create_individual(Node, Genome))

    return pop


def _create_individual(Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
                       Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]]) -> \
        List[Union[ModuleGenome, BlueprintGenome, DAGenome]]:
    in_node_params = (0, NodeType.INPUT)
    out_node_params = (1, NodeType.OUTPUT)
    mid_node_params = (2, NodeType.HIDDEN)

    linear = Genome(
        ([Node(*in_node_params), Node(*mid_node_params), Node(*out_node_params)]),
        [Connection(1, 0, 2), Connection(2, 2, 1)]  # TODO should this have the connection from 0 -> 1?
    )

    tri = Genome(
        ([Node(*in_node_params), Node(*mid_node_params), Node(*out_node_params)]),
        [Connection(0, 0, 1), Connection(1, 0, 2), Connection(2, 2, 1)]
    )

    dia = Genome(
        ([Node(*in_node_params), Node(*mid_node_params), Node(3, NodeType.HIDDEN), Node(*out_node_params)]),
        [Connection(1, 0, 2), Connection(3, 0, 3), Connection(4, 3, 1), Connection(2, 2, 1)]
    )

    # Making the in and out nodes of modules blank
    genomes = [linear, tri, dia]
    if (config.blank_module_input_nodes and Node == ModuleNode) or (config.blank_bp_input_nodes and Node == BlueprintNode):
        for genome in genomes:
            genome.nodes[0] = _blank_node(genome.get_input_node())  # 0 is always input

    if (config.blank_module_output_nodes and Node == ModuleNode) or (config.blank_bp_output_nodes and Node == BlueprintNode):
        for genome in genomes:
            genome.nodes[1] = _blank_node(genome.get_output_node())  # 1 is always output

    return genomes


def create_mr() -> MutationRecords:
    return MutationRecords({(0, 1): 0, (0, 2): 1, (2, 1): 2, (0, 3): 3, (3, 1): 4},
                           {(0, 0): 2, (0, 1): 3},
                           3, 4)


def _blank_node(node: Union[ModuleNode, BlueprintNode, DANode]) -> ModuleNode:
    """Makes a module node that only return its input and doesn't allow it to change"""
    new_node = ModuleNode(node.id, node.node_type)

    new_node.layer_type.set_value(None)
    dropout = new_node.layer_type.get_submutagen('dropout')
    regularisation = new_node.layer_type.get_submutagen('regularisation')
    dropout.set_value(None)
    regularisation.set_value(None)
    new_node.layer_type.mutation_chance = 0
    dropout.mutation_chance = 0
    regularisation.mutation_chance = 0

    return new_node
