# Analysis tools guide

EvolutionaryDataPlotter, GraphPlotter, and TrainingResultsReader 
are all executable

to use TrainingResultsReader, assuming you have conformed to the results files 
conventions, simply run it, and it will produce a formatted message with each of 
the max scores of each fully trained network which has been run

to use EvolutionaryDataPlotter or GraphPlotter, copy the desired run 
folders out of results/dataset/evolution/configuration and into data/runs.
Whichever runs folders are in data/runs will be used by EvolutionaryDataPlotter 
and GraphPlotter when they are run


EvolutionaryDataPlotter is our main exploratory tool for the performance of 
different run configurations during evolution. It has a number of plotting 
options which include: 

    aggregation_type: the type of aggregation to use to reduce a given 
    generations scores to one score
    
    show_smoothed_data, smooth_boundries: whether or not to use rolling 
    average data smoothing
    
    show_best_fit: to plot the best fit lines of runs
    
    stay_at_max: to plot run performances without showing drops (stay at max), 
    this option plots the rolling max score
    
    smooth_boundries: to show run group performance boundries. A run group 
    is a set of runs with the same config, ie: x_1...x_5
    
    colour_group_run_lines_same: to make all runs from the same run group 
    (in the same boundry) visibly indistinguishable, to make comparitive plots
    between groups more clear
