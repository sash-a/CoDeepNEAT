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

### Speciation improvements

See Sasha's paper

#### Config options
