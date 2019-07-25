def validate_fold(module_graph, dataset,  k=10, i=0):
    pass

def validate_standard(module_graph, dataset):



def create_nn(module_graph):

    if module_graph is None:
        raise Exception("None module graph produced from blueprint")
    try:
        net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs)).to(device)

    except Exception as e:
        if Config.save_failed_graphs:
            module_graph.plot_tree_with_graphvis("Module graph which failed to parse to nn")
        raise Exception("Error: failed to parse module graph into nn", e)