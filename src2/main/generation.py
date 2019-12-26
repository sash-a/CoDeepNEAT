"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""
from __future__ import annotations

import torch.multiprocessing as mp
from typing import Optional, List

import wandb

from runs import runs_manager

from src2.utils.wandb_utils import wandb_log
from src2.utils.mp_utils import get_bp_eval_pool
from src2.configuration import config
from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src2.genotype.cdn.genomes.da_genome import DAGenome
from src2.genotype.cdn.genomes.module_genome import ModuleGenome
from src2.genotype.cdn.mutators.blueprint_genome_mutator import BlueprintGenomeMutator
from src2.genotype.cdn.mutators.module_genome_mutator import ModuleGenomeMutator
from src2.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src2.genotype.cdn.nodes.da_node import DANode
from src2.genotype.cdn.nodes.module_node import ModuleNode
from src2.genotype.cdn.population_initializer import create_population, create_mr
from src2.genotype.neat.operators.speciators.most_similar_speciator import MostSimilarSpeciator
from src2.genotype.neat.operators.speciators.neat_speciator import NEATSpeciator
from src2.genotype.neat.population import Population
from src2.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src2.phenotype.neural_network.neural_network import Network
from src2.phenotype.phenotype_evaluator import evaluate_blueprints
import src2.main.singleton as singleton


class Generation:
    def __init__(self):
        self.genome_id_counter = 0  # max genome id of all genomes contained in this generation obj
        singleton.instance = self

        self.module_population: Optional[Population] = None
        self.blueprint_population: Optional[Population] = None
        self.da_population: Optional[Population] = None

        self.initialise_populations()
        self.generation_number = 0

    def step_evaluation(self):
        self.evaluate_blueprints()

        # Aggregate the fitness immediately after they have all been recorded
        self.module_population.aggregate_fitness()
        self.blueprint_population.aggregate_fitness()

        best = self.blueprint_population.get_most_accurate()
        print("Best nn: {} - {:.2f}%".format(best.id, best.accuracy * 100))

    def step_evolution(self):
        """Runs cdn for one generation. Prepares population objects for the next step."""
        if config.use_wandb:
            wandb_log(self)

        # TODO move this to visualization method
        most_accurate_blueprint: BlueprintGenome = self.blueprint_population.get_most_accurate()
        if config.plot_best_genotypes:
            most_accurate_blueprint.visualize(prefix="best_g" + str(self.generation_number) + "_")
        if config.plot_best_phenotype:
            model: Network = Network(most_accurate_blueprint, get_data_shape(),
                                     sample_map=most_accurate_blueprint.best_module_sample_map)
            model.visualize(prefix="best_g" + str(self.generation_number) + "_")

        self.module_population.step()
        self.blueprint_population.step()

        # self.module_population.visualise(suffix="_" + str(self.generation_number) + "_module_species")
        # self.blueprint_population.visualise(suffix="_" + str(self.generation_number) + "blueprint_species")

        self.generation_number += 1

        self.module_population.end_step()
        self.blueprint_population.end_step()
        if config.evolve_data_augmentations:
            self.da_population.end_step()

        print('Module species:', [len(spc.members) for spc in self.module_population.species])
        print('Step ended\n\n')

    def evaluate_blueprints(self):
        """Evaluates all blueprints"""
        self.module_population.before_step()
        self.blueprint_population.before_step()
        # blueprints choosing DA schemes from population
        if config.evolve_data_augmentations:
            self.da_population.before_step()
            for blueprint_individual in self.blueprint_population:
                blueprint_individual.pick_da_scheme()

        # Kicking off evaluation
        in_size = get_data_shape()
        consumable_q = mp.get_context('spawn').Manager().Queue(len(list(self.blueprint_population)))

        for bp in self.blueprint_population:
            consumable_q.put(bp, False)

        with get_bp_eval_pool(self) as pool:  # TODO will probably be more efficient to keep this alive throughout gens
            futures = []
            for i in range(config.n_gpus * config.n_evals_per_gpu):
                futures.append(pool.submit(evaluate_blueprints, consumable_q, in_size))

            self.report_fitness(futures)

    def report_fitness(self, futures):
        """
        Collects the results from each process and assigns them to the blueprints and modules on this process
        """
        results: List[BlueprintGenome] = []
        for future in futures:
            results += future.result()

        # Replacing the bp population with the blueprints that are returned from the processes
        # i.e the same blueprints, but they have fitness assigned
        self.blueprint_population.species[0].members = {bp.id: bp for bp in results}
        # Reporting fitness to all modules
        for bp in self.blueprint_population:
            for fitness, sample_map in zip(bp.fitness_raw[0], bp.all_sample_maps):
                bp.report_fitness_to_modules([fitness], sample_map)

    def initialise_populations(self):
        """Starts off the populations of a new evolutionary run"""
        if config.module_speciation.lower() == "similar":
            module_speciator = MostSimilarSpeciator(config.species_distance_thresh_mod_base, config.n_module_species,
                                                    ModuleGenomeMutator())
        elif config.module_speciation.lower() == "neat":
            module_speciator = NEATSpeciator(config.species_distance_thresh_mod_base, config.n_module_species,
                                             ModuleGenomeMutator())
        else:
            raise Exception(
                "speciation method in config not recognised: " + str(config.module_speciation).lower()
                + " expected: similar | neat")

        bp_speciator = NEATSpeciator(config.species_distance_thresh_mod_base, config.n_blueprint_species,
                                     BlueprintGenomeMutator())

        self.module_population = Population(create_population(config.module_pop_size, ModuleNode, ModuleGenome),
                                            create_mr(), config.module_pop_size, module_speciator)

        self.blueprint_population = Population(
            create_population(config.bp_pop_size, BlueprintNode, BlueprintGenome),
            create_mr(), config.bp_pop_size, bp_speciator)

        print("initialised pops, bps:", len(self.blueprint_population), "mods:", len(self.module_population))

        # TODO DA pop only straight genomes
        if config.evolve_data_augmentations:
            self.da_population = Population(create_population(config.da_pop_size, DANode, DAGenome),
                                            create_mr(), config.da_pop_size, bp_speciator)

    def __getitem__(self, genome_id: int):
        if config.evolve_data_augmentations:
            populations: List[Population] = [self.blueprint_population, self.module_population, self.da_population]
        else:
            populations: List[Population] = [self.blueprint_population, self.module_population]

        for pop in populations:
            mem = pop[genome_id]
            if mem is not None:
                return mem

        return None
