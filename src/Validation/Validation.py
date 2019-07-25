from src.Config import Config

def validate_fold(module_graph, dataset,  k=10, i=0):
    pass

def create_nn(module_graph, inputs):
    blueprint_individual = module_graph.self.blueprint_genome

    if module_graph is None:
        raise Exception("None module graph produced from blueprint")
    try:
        net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs)).to(Config.get_device())

    except Exception as e:
        if Config.save_failed_graphs:
            module_graph.plot_tree_with_graphvis("Module graph which failed to parse to nn")
        raise Exception("Error: failed to parse module graph into nn", e)


    net.configure(blueprint_individual.learning_rate(), blueprint_individual.beta1(), blueprint_individual.beta2())
    net.specify_dimensionality(inputs)

    return net