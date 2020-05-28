from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from src.genotype.neat.operators.representative_selectors.representative_selector import RepresentativeSelector

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class BestRepSelector(RepresentativeSelector):
    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        return max(genomes.values(), key=lambda genome: genome.aggregated_acc)
