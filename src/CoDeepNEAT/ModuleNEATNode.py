from src.NEAT.NEATNode import NEATNode
from src.NEAT.NEATNode import NodeType
from src.CoDeepNEAT.Mutagen import Mutagen
from src.CoDeepNEAT.Mutagen import ValueType
import torch.nn as nn
import torch
import torch.nn.functional as F


class ModulenNEATNode(NEATNode):

    def __init__(self, id, x, node_type=NodeType.HIDDEN,
                 out_features=25, activation=F.relu, layer_type=nn.Conv2d,
                 conv_window_size=-1, conv_stride=1, regularisation=None, reduction="MaxPool2d", max_pool_size = -1):
        super(ModulenNEATNode, self).__init__(id, x, node_type)


        self.activation = Mutagen(F.relu, F.leaky_relu, F.sigmoid, F.relu6)  # TODO try add in Selu, Elu
        self.activation.set_value(activation)

        self.out_features = Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=out_features, start_range=1,
                                    end_range=100)

        self.layer_type = Mutagen(nn.Conv2d, nn.Linear, sub_mutagens= {
            nn.Conv2d: {"conv_window_size": Mutagen(3,5,7), "conv_stride": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=conv_stride, start_range=1,end_range=5)}
        })
        self.layer_type.set_value(layer_type)
        self.layer_type.set_sub_value(nn.Conv2d, "conv_window_size", conv_window_size)

        self.reduction = Mutagen(None, nn.MaxPool2d, sub_mutagens={
            nn.MaxPool2d:{"pool_size":Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=max_pool_size, start_range=2,end_range=5)}})
        self.reduction.set_value(nn.MaxPool2d)

        self.regularisation = Mutagen(None, nn.BatchNorm2d)
        self.regularisation.set_value(nn.BatchNorm2d)
