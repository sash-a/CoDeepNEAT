from __future__ import annotations

from typing import Union, Type, TYPE_CHECKING, List

from configuration import config
from src.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src.genotype.cdn.nodes.module_node import ModuleNode

from .shapes import str_to_shape, str_to_shape_mr
from ...neat.mutation_record import MutationRecords

from src.genotype.cdn.genomes.module_genome import ModuleGenome
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome

if TYPE_CHECKING:
    from src.genotype.cdn.genomes.da_genome import DAGenome
    from src.genotype.cdn.nodes.da_node import DANode


def create_population(pop_size: int, Node: Union[Type[ModuleNode], Type[BlueprintNode], Type[DANode]],
                      Genome: Union[Type[ModuleGenome], Type[BlueprintGenome], Type[DAGenome]],
                      *shapes: str) -> \
        List[Union[ModuleGenome, BlueprintGenome, DAGenome]]:
    """Creates a population of the given types and initializes it with the given shapes"""
    validate_shapes(shapes)

    pop = []
    while len(pop) < pop_size:
        for shape in shapes:
            pop.append(str_to_shape[shape](Node, Genome))
            _blank_genome_nodes(pop[-1])

    pop = pop[:pop_size]
    return pop


def create_mr(*shapes: str):
    """Creates the mutation record for the given shapes"""
    validate_shapes(shapes)
    mr = MutationRecords({}, {}, 1, 0)
    for shape in shapes:
        str_to_shape_mr[shape](mr)

    return mr


def _blank_genome_nodes(genome: Union[ModuleGenome, BlueprintGenome, DAGenome]):
    """Blanks a genomes nodes if the config options allow it"""
    if (config.blank_module_input_nodes and isinstance(genome, ModuleGenome)) or \
            (config.blank_bp_input_nodes and isinstance(genome, BlueprintGenome)):
        genome.nodes[0] = _blank_node(genome.get_input_node())  # 0 is always input

    if (config.blank_module_output_nodes and isinstance(genome, ModuleGenome)) or \
            (config.blank_bp_output_nodes and isinstance(genome, BlueprintGenome)):
        genome.nodes[1] = _blank_node(genome.get_output_node())  # 1 is always output


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


def validate_shapes(shapes):
    """Verifies sure all shapes are valid"""
    if len(shapes) == 0:
        raise ValueError(f'Must have at least one shape. Valid shapes are: {str_to_shape.keys()}')

    for shape in shapes:
        if shape not in str_to_shape:
            raise ValueError(f'{shape} is not a valid shape. Valid shapes are: {str_to_shape.keys()}')
