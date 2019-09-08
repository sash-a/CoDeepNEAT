import graphviz

from src.Config import Config
from src.NeuralNetwork.ModuleNet import ModuleNet
from src.Phenotype.ModuleNode import ModuleNode


class ModuleGraph():

    """represents the whole module graph. simply holds a handle on the root module node"""

    def __init__(self, module_graph_root_node, dataset=""):

        self.module_graph_root_node: ModuleNode = module_graph_root_node
        if dataset == "":
            self.dataset = Config.dataset

        self.blueprint_genome = None
        self.fitness_values = []
        self.data_augmentation_schemes = []

    def to_nn(self, in_features):
        """converts this full dnn module graph into a dnn"""
        self.module_graph_root_node.create_layer(in_features)
        return ModuleNet(self)

    def get_net_size(self):
        """returns the complexity of the full dnn, ie the blueprints complexity"""
        net_params = self.module_graph_root_node.get_parameters({})
        return sum(p.numel() for p in net_params if p.requires_grad)

    def delete_all_layers(self):
        """flashes the weights, to allow for compact saving"""
        for node in self.module_graph_root_node.get_all_nodes_via_bottom_up(set()):
            node.delete_layer()

    def plot_tree_with_graphvis(self, title="", file="temp", view=None, graph=None, return_graph_obj=False):
        if graph is None:
            graph = graphviz.Digraph(comment=title)

        self.module_graph_root_node.plot_tree_with_graphvis(title=title, file=file, graph=graph)

        for da_scheme in self.data_augmentation_schemes:
            da_scheme.plot_tree_with_graphvis(title=title, file=file, graph=graph)

        graph.render(file, view=view)

        if return_graph_obj:
            return graph
