import copy
import heapq
import random

from src.CoDeepNEAT.CDNNodes.ModuleNode import ModuleNEATNode
from src.Config import Config, NeatProperties as Props
from src.NEAT.Gene import NodeGene, NodeType
from src.NEAT.Mutagen import Mutagen, ValueType


class BlueprintNEATNode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN, representative=None):
        super(BlueprintNEATNode, self).__init__(id, node_type)

        self.species_number = Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=0, start_range=0,
                                      end_range=1, print_when_mutating=False, name="species number",
                                      mutation_chance=0.5, inherit_as_discrete=True)

        self.chosen_module: ModuleNEATNode = None
        self.target_num_species_reached = False

        # Changes blueprint from using species indexes to using representatives, allowing the node to act as a species
        if Config.blueprint_nodes_use_representatives:
            self.representative = representative

    def get_similar_modules(self, modules, n):
        """:returns the n most similar modules to to self.representative"""
        if not Config.blueprint_nodes_use_representatives:
            raise Exception('get_similar_modules called, but use representatives is false')

        return heapq.nsmallest(n, modules, key=lambda indv: indv.distance_to(self.representative))

    def choose_representative(self, modules, all_reps):
        """Chooses a representative for self"""
        all_reps = list(set(all_reps))  # removing duplicated to make choosing fair
        chance = random.random()
        # If rep is none ignore chance to pick similar rep
        chance_pick_rand = 0.7
        if self.representative is None:
            chance_pick_rand = 1

        if chance < 0.5 and all_reps:
            # 50% chance to pick random from reps already in the blueprint to promote repeating structures
            self.representative = random.choice(all_reps)
        elif chance < chance_pick_rand:
            # 20% or 50% chance to pick random from pop
            new_rep = copy.deepcopy(random.choice(modules))

            for rep in all_reps:
                if new_rep == rep:
                    new_rep = rep
                    break

            self.representative = new_rep
        elif chance < 0.75:
            # 0% or 5% chance to pick a very different representative
            new_rep = copy.deepcopy(
                random.choice(heapq.nlargest(10, modules, key=lambda indv: indv.distance_to(self.representative))))

            for rep in all_reps:
                if new_rep == rep:
                    new_rep = rep
                    break

            self.representative = new_rep
        else:
            # 0% or 20% chance to pick a similar representative
            choices = self.get_similar_modules(modules, Config.closest_reps_to_consider)

            weights = [2 - (x / Config.closest_reps_to_consider) for x in
                       range(Config.closest_reps_to_consider)]  # closer reps have a higher chanecs
            self.representative = random.choices(choices, weights=weights, k=1)[0]

        return self.representative

    def get_all_mutagens(self):
        return [self.species_number]

    def set_species_upper_bound(self, num_species, generation_number):
        """used to update the species number mutagens. takes care of the species number shuffling"""
        if not self.target_num_species_reached and num_species >= Props.MODULE_TARGET_NUM_SPECIES:
            """species count starts low, and increases quickly. 
            due to low species number mutation rates, nodes would largely be stuck
            referencing species 0 for many generations before a good distribution arises
            so we force a shuffle in the early generations, to get a good distribution early"""
            self.target_num_species_reached = True
            self.species_number.mutation_chance = 0.13  # the desired stable mutation rate
            if generation_number < 3:
                """by now, the target species numbers have been reached, but the high mutation rate \
                has not had enough time to create a good distribution. so we shuffle species numbers"""
                self.species_number.set_value(random.randint(0, num_species - 1))

        self.species_number.end_range = num_species
        if self.species_number() >= num_species:
            self.species_number.set_value(random.randint(0, num_species - 1))

    def get_node_name(self):
        return "Species:" + repr(self.species_number())

    def pick_module(self, species_module_index_map, module_species):
        if self.species_number.value in species_module_index_map:
            mod_idx = species_module_index_map[self.species_number.value]
            if isinstance(mod_idx, tuple):
                spc, mod = mod_idx
                module_genome = module_species[spc][mod]
            else:
                module_genome = module_species[self.species_number.value][mod_idx]
        else:
            module_genome, mod_idx = module_species[self.species_number.value].sample_individual()
            species_module_index_map[self.species_number.value] = mod_idx

        return module_genome
