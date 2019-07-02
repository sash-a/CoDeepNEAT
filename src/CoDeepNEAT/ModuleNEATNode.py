from src.NEAT.NEATNode import NEATNode
from src.NEAT.NEATNode import NodeType
from src.NEAT.Mutagen import Mutagen
from src.NEAT.Mutagen import ValueType
import torch.nn as nn
import torch.nn.functional as F
import torch


class ModulenNEATNode(NEATNode):

    def __init__(self, id, x, node_type=NodeType.HIDDEN,
                 out_features=25, activation=F.relu, layer_type=nn.Conv2d,
                 conv_window_size=7, conv_stride=1, regularisation=None, reduction=nn.MaxPool2d, max_pool_size = 2):
        super(ModulenNEATNode, self).__init__(id, x, node_type)


        self.activation = Mutagen(F.relu, F.leaky_relu, torch.sigmoid, F.relu6, discreet_value= activation)  # TODO try add in Selu, Elu

        self.out_features = Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=out_features, start_range=1,
                                    end_range=100)

    #################force linear
        self.layer_type = Mutagen(nn.Linear, discreet_value= nn.Linear, sub_mutagens= {
            nn.Linear:{"regularisation":Mutagen(None, nn.BatchNorm1d, discreet_value= None), "reduction": Mutagen(None, discreet_value=None)}
        })
    #################force convolution
        # self.layer_type = Mutagen(nn.Conv2d, discreet_value= nn.Conv2d,
        #                           sub_mutagens= {nn.Conv2d: {"conv_window_size": Mutagen(3,5,7, discreet_value=conv_window_size), "conv_stride": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=conv_stride, start_range=1,end_range=5),
        #                                                      "reduction": Mutagen(None, nn.MaxPool2d, discreet_value=None , sub_mutagens= {nn.MaxPool2d:{"pool_size":Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=max_pool_size, start_range=2,end_range=5)}}),
        #                                                      "regularisation": Mutagen(None, nn.BatchNorm2d, discreet_value=None)
        #                                                      }})
    #################use both
        # self.layer_type = Mutagen(None, nn.MaxPool2d, sub_mutagens={
        #     nn.MaxPool2d:{"pool_size":Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=max_pool_size, start_range=2,end_range=5)}}
        #                          , discreet_value=reduction)


    def get_all_mutagens(self):
        return [self.activation,self.out_features,self.layer_type]