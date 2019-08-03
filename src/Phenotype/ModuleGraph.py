from src.Config.Config import Config
from src.NeuralNetwork.ModuleNet import ModuleNet
from src.Phenotype.ModuleNode import ModuleNode


class ModuleGraph():

    def __init__(self, module_graph_root_node, dataset=""):

        self.module_graph_root_node: ModuleNode = module_graph_root_node
        if dataset == "":
            self.dataset = Config.dataset

        self.blueprint_genome = None
        self.fitness_values = []
        self.data_augmentation_schemes = []

    def to_nn(self, in_features):
        self.module_graph_root_node.create_layer(in_features)
        return ModuleNet(self)

    def get_net_size(self):
        net_params = self.module_graph_root_node.get_parameters({})
        return sum(p.numel() for p in net_params if p.requires_grad)

    def delete_all_layers(self):
        for node in self.module_graph_root_node.get_all_nodes_via_bottom_up(set()):
            node.delete_layer()

    def plot_tree_with_graphvis(self, title="", file="temp"):
        self.module_graph_root_node.plot_tree_with_graphvis(title=title, file=file)
