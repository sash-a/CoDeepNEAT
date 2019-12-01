from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, List

import math

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Species import Species

from graphviz import Graph

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from graphviz import Digraph


def get_graph_from_species(species: Species, sub_graph=False):
    name = ("cluster_" if sub_graph else "") + "species_" + str(species.id)
    species_graph = Digraph(name=name)

    mems_graph = Digraph(name="cluster_mems_from_species_" + str(species.id) + "species")

    i = 0
    r = 1
    for mem in species.members.keys():
        r += 1
        x = math.cos(i) * r
        y = math.sin(i) * r
        mems_graph.node(name=str(mem), pos=str(x) + "," + str(y))
        i += 1

    mems = list(species.members.keys())
    for m in range(len(mems)):
        for mem2 in species.members.keys():
            if mem == mem2:
                continue
            if random.random() < 1/len(species.members):
                mems_graph.edge(str(mem), str(mem2))

        mems_graph.edge(str(mems[m]) , str(mems[(m+1)%len(mems)]))

    species_graph.subgraph(mems_graph)

    return species_graph


def visualise_specieses(specieses: List[Species], suffix=""):
    generation_graph = Digraph(name="gen" + suffix)
    for spc in specieses:
        generation_graph.subgraph(get_graph_from_species(spc, sub_graph=True))

    generation_graph.view()
