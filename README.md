# CoDeepNEAT

An implementation of CoDeepNEAT with extensions

## Setup

```conda create -n cdn --file requirements.txt``` to install all requirements

```conda activate cdn``` to activate the env

Evolution environment is the main entry point run this file simply with:

```python src/EvolutionEnvironment/EvolutionEnvironment.py```

All config options are in src/Config/ directory

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

```adjust_species_mutation_magnitude_based_on_fitness```: an experimental extension which controls 
species relative mutation rates based on their fitnesses

```adjust_mutation_magnitudes_over_run```: the global mutation magnitude extension option

```breed_mutagens```: toggles whether or not to breed mutagens/attributes during crossover

### Speciation improvements

See Sasha's paper

#### Config options


### Other extension options

```allow_elite_cloning```: experimental. allows for asexual reproduction of elite blueprints