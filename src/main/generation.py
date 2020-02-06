"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""
from __future__ import annotations

import torch.multiprocessing as mp
from typing import Optional, List

from src.genotype.neat.operators.population_rankers.single_objective_rank import SingleObjectiveRank
from src.genotype.neat.operators.population_rankers.two_objective_rank import TwoObjectiveRank
from src.utils.mp_utils import get_bp_eval_pool
from src.configuration import config
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.genotype.cdn.genomes.da_genome import DAGenome
from src.genotype.cdn.genomes.module_genome import ModuleGenome
from src.genotype.cdn.mutators.blueprint_genome_mutator import BlueprintGenomeMutator
from src.genotype.cdn.mutators.da_genome_mutator import DAGenomeMutator
from src.genotype.cdn.mutators.module_genome_mutator import ModuleGenomeMutator
from src.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src.genotype.cdn.nodes.da_node import DANode
from src.genotype.cdn.nodes.module_node import ModuleNode
from src.genotype.cdn.population_initializer import create_population, create_mr
from src.genotype.neat.operators.speciators.most_similar_speciator import MostSimilarSpeciator
from src.genotype.neat.operators.speciators.neat_speciator import NEATSpeciator
from src.genotype.neat.population import Population
from src.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src.phenotype.neural_network.neural_network import Network
from src.phenotype.phenotype_evaluator import evaluate_blueprints
import src.main.singleton as singleton


class Generation:
    def __init__(self):
        self.genome_id_counter = 0  # max genome id of all genomes contained in this generation obj
        singleton.instance = self

        self.module_population: Optional[Population] = None
        self.blueprint_population: Optional[Population] = None
        self.da_population: Optional[Population] = None

        self.init_populations()
        self.generation_number = 0

    def step_evaluation(self):
        self.evaluate_blueprints()

        # Aggregate the fitness immediately after they have all been recorded
        self.module_population.aggregate_fitness()
        self.blueprint_population.aggregate_fitness()

        best = self.blueprint_population.get_most_accurate()[0]
        print("Best nn: {} - {:.2f}%".format(best.id, best.accuracy * 100))

    def step_evolution(self):
        """Runs cdn for one generation. Prepares population objects for the next step."""
        # TODO move this to visualization method
        most_accurate_blueprint: BlueprintGenome = self.blueprint_population.get_most_accurate()[0]
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
        if config.evolve_da and config.evolve_da_pop:
            self.da_population.end_step()

        print('Module species:', [len(spc.members) for spc in self.module_population.species])
        print('Step ended\n\n')

    def evaluate_blueprints(self):
        """Evaluates all blueprints"""
        self.module_population.before_step()
        self.blueprint_population.before_step()
        # blueprints choosing DA schemes from population
        if config.evolve_da and config.evolve_da_pop:
            self.da_population.before_step()
            for blueprint_individual in self.blueprint_population:
                blueprint_individual.sample_da()

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
        bp: BlueprintGenome
        for bp in self.blueprint_population:
            accuracies = bp.fitness_raw[0]
            for accuracy, sample_map in zip(accuracies, bp.all_sample_maps):
                bp.report_module_fitness(accuracy, sample_map)
                if config.evolve_da and config.evolve_da_pop:
                    bp.report_da_fitness(accuracy)

    def init_populations(self):
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

        if not config.multiobjective:
            ranker = SingleObjectiveRank()
        else:
            ranker = TwoObjectiveRank()

        self.module_population = Population(create_population(config.module_pop_size, ModuleNode, ModuleGenome),
                                            create_mr(), config.module_pop_size, module_speciator, ranker)

        self.blueprint_population = Population(
            create_population(config.bp_pop_size, BlueprintNode, BlueprintGenome),
            create_mr(), config.bp_pop_size, bp_speciator, ranker)

        print("initialised pops, bps:", len(self.blueprint_population), "mods:", len(self.module_population))

        # TODO DA pop only straight genomes
        if config.evolve_da and config.evolve_da_pop:
            da_speciator = NEATSpeciator(config.species_distance_thresh_mod_base, config.n_blueprint_species,
                                         DAGenomeMutator())
            self.da_population = Population(create_population(config.da_pop_size, DANode, DAGenome, no_branches=True),
                                            create_mr(), config.da_pop_size, da_speciator, SingleObjectiveRank())

    def __getitem__(self, genome_id: int):
        if config.evolve_da and config.evolve_da_pop:
            populations: List[Population] = [self.blueprint_population, self.module_population, self.da_population]
        else:
            populations: List[Population] = [self.blueprint_population, self.module_population]

        for pop in populations:
            if pop is None:
                continue

            mem = pop[genome_id]
            if mem is not None:
                return mem

        return None


def get_num_objectives_for(genome):
    if isinstance(genome, BlueprintGenome) or isinstance(genome, ModuleGenome):
        return 2
    if isinstance(genome, DAGenome):
        return 1
    raise ValueError()