from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import List, Union, TYPE_CHECKING
from src.genotype.neat.operators.mutators.mutation_report import MutationReport

if TYPE_CHECKING:
    from src.genotype.cdn.nodes.blueprint_node import BlueprintNode
    from src.genotype.cdn.nodes.da_node import DANode
    from src.genotype.cdn.nodes.module_node import ModuleNode
    from src.genotype.neat.connection import Connection
    from src.genotype.mutagen.mutagen import Mutagen


class Gene(ABC):
    """base class for a neat node and connection"""

    def __init__(self, id: int):
        self.id: int = id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    @abstractmethod
    def get_all_mutagens(self) -> List[Mutagen]:
        raise NotImplementedError('get_all_mutagens should not be called from Gene')

    def mutate(self) -> MutationReport:
        report = MutationReport()
        for mutagen in self.get_all_mutagens():
            report += mutagen.mutate()
        return report

    def interpolate(self, other: Union[ModuleNode, BlueprintNode, DANode, Connection]):
        child = copy.deepcopy(self)
        for (best, worst) in zip(child.get_all_mutagens(), other.get_all_mutagens()):
            best.interpolate(worst)

        return child
