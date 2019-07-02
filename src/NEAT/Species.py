import src.NEAT.NeatProperties as Props
from src.NEAT.Genome import Genome
from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
from random import randint

from Multiobjective.ParetoFront import CDN_pareto

from typing import List, Union


class Species:

    def __init__(self, representative: Union[Genome, ModuleGenome, BlueprintGenome]):
        self.representative: Union[Genome, ModuleGenome, BlueprintGenome] = representative
        self.members: List[Union[Genome, ModuleGenome, BlueprintGenome]] = [representative]

    def add_member_safe(self, new_member, thresh=Props.SPECIES_DISTANCE_THRESH):
        if self.is_compatible(new_member, thresh):
            self.members.append(new_member)

    def is_compatible(self, individual, thresh=Props.SPECIES_DISTANCE_THRESH, c1=1, c2=1):
        return self.representative.distance_to(individual, c1, c2) <= thresh

    def clear(self):
        self.members.clear()

    def sample_individual(self):
        index = randint(0, len(self.members) - 1)
        return self.members[index], index

    def pareto_front(self, algorithm='cdn'):
        if not (type(self.representative) == ModuleGenome or type(self.representative) == BlueprintGenome):
            raise TypeError('Genome is not multiobjective, therefore cannot find the pareto front')

        if algorithm == 'cdn':
            return CDN_pareto(self.members)
        else:
            raise ValueError('algorithm may only be \'cdn\'')
