# CoDeepNEAT

An implementation of implementation of CoDeepNEAT, originally created by Risto Miikkulainen et al. with our own extensions. Implementation details were taken from their [2017](https://arxiv.org/pdf/1703.00548/) and [2019](https://arxiv.org/pdf/1902.06827.pdf) paper.

## Setup
Requires [conda](https://docs.conda.io/en/latest/) 
```
conda create -n cdn --file requirements.txt
conda activate cdn
pip install tarjan wandb  # these are not available from conda
```  

## Entry points
Directory: ```src/main/```  
```ft.py``` Fully trains a run from evo.py  
```evo.py``` Does an evolutionary run  
```batch_run.py``` Running many different configurations all the way from evolution to fully training. (See note below)  

## Config
All config options are in ```src/configuration/configuration.py```  
Example configs are in ```src/configuration/configs``` directory  

## How to run
```python src/main/evo.py -g 1 -c base```

## Extensions
Extensions are detailed in the papers inside the papers directory

#### Note about batch runs
This system was developed for rapid tuning of CDN's own hyperparameters on a cluster with a limited number of GPUs. It should not be used for normal training as it was created for our very specific case. Rather do a single run on ```evo.py``` and then fully train it with ```ft.py```.