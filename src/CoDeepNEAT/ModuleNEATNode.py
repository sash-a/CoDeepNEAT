from src.NEAT.NEATNode import NEATNode
from src.NEAT.NEATNode import NodeType

class ModulenNEATNode(NEATNode):


    def __init__(self, id, x, out_features = 25, activation = "relu" , node_type = NodeType.HIDDEN, layer_type = "Conv2d", conv_window_size = -1, conv_stride = 1, regularisation = None, reduction = "MaxPool2d"):

        super(ModulenNEATNode, self).__init__(id,x,node_type )
        self.layer_type = layer_type
        self.regularisation = regularisation
        self.reduction = reduction
        self.out_featrures = out_features
        self.activation = activation

        if(self.layer_type == "Conv2d"):
            self.conv_window_size = conv_window_size
            self.conv_stride = conv_stride





