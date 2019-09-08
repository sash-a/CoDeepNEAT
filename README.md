# CoDeepNEAT

An implementation of CoDeepNEAT with extensions

## Setup

```conda create -n cdn --file requirements.txt``` to install all requirements

```conda activate cdn``` to activate the env

Evolution environment is the main entry point run this file simply with:

```python src/EvolutionEnvironment/EvolutionEnvironment.py```

All config options are in ```src/Config/``` directory

#### General config options

```continue_from_last_run```: continue from a run with the same name as stored in ```run_name``` located at ```data/runs/run_name```

```num_gpus```: the number of GPUs available

```dataset```: the dataset to be used, current options are: mnist, fashion_mnist, cifar10

```data_path```: path to the datasets root folder, leave blank for data/Datasets/

```number_of_epochs_per_evaluation```: epochs to train for during evolution

```second_objective```: the name of the second objective, current options are network_size, network_size_adjusted

```second_objective_comparator```: how the second objectives should be optimized e.g ```operator.lt``` for minimisation

## Extensions:

### Data augmentations

See Liron's paper

#### Config options
```evolve_data_augmentations```: turns on/off the data augmentations extension

```colour_augmentations```: determines if photometric (colour-based) augmentations are included in the evolvable population (cannot be used with black and white images) (see appendix B in Liron's paper)

```allow_da_scheme_ignores```: turns on/off the DA_scheme_ignores modification (see section 4.2 in Liron's paper)

```train_on_original_data```: determines if evolved networks train on both the original data and augmented data (reccomended) or just the augmented data

```batch_by_batch```: determines how networks train on data. Networks can train batch by batch, meaning they train on one batch of original data and then one batch of augmented data or epoch by epoch (reccomended), in which case they train on original data for one epoch and then augmented data for the next.

### Elitism improvements

See Shane's paper

#### Config options

```module_retention```: determines if the module retention option is active

```fitness_aggregation```: option between max/avg - decides what type of aggregation cdn individuals use for their fitnesses

```allow_species_module_mapping_ignores```: allows the chance to ignore some of the blueprint species to module mappings

```allow_cross_species_mappings```: decides if mappings persist when mapped modules change to different species

### Mutation extensions

#### Config options

```adjust_species_mutation_magnitude_based_on_fitness```: an experimental extension which controls species relative mutation rates based on their fitnesses

```adjust_mutation_magnitudes_over_run```: the global mutation magnitude extension option

```breed_mutagens```: toggles whether or not to breed mutagens/attributes during crossover

### Speciation improvements

See Sasha's paper

#### Config options

```speciation_overhaul```: turns off traditional NEAT speciation and replaces it with best representative selection (Sasha's paper section 3.2.1)

```use_graph_edit_distance```: replaces NEAT distance with graph edit distance when comparing topological similarity (Sasha's paper section 3.2.2)

```allow_attribute_distance```: add attribute distance to topological distance when finding the similarity of individuals (Sasha's paper section 3.2.3) For more fine grained control see ```LAYER_SIZE_COEFFICIENT``` and ```LAYER_TYPE_COEFFICIENT``` in ```src/Config/NeatProperties```

### Representative overhaul

#### Config options

```blueprint_nodes_use_representatives```: uses representatives instead of species numbers at the blueprint node level (Sasha's paper section 3.2.4)

```rep_mutation_chance_early```: chance to mutate representatives in the first 3 generations

```rep_mutation_chance_late```: chance to mutate representatives after 3 generations

```similar_rep_mutation_chance```: chance to mutate all of the nodes with the same representative in the same way

```closest_reps_to_consider```: number of individuals to choose from when finding the closest individuals

### Other extension options

```allow_elite_cloning```: experimental. allows for asexual reproduction of elite blueprints