from torch import nn

from src.CoDeepNEAT.CDNNodes import ModulenNEATNode
import src.Config.Config as Config

class Layer(nn.Module):
    def __init__(self, module: ModulenNEATNode, feature_multiplier=1):
        super().__init__()
        device = Config.get_device()

        self.module_node: ModulenNEATNode = module

        self.out_features = round(module.layer_type.get_sub_value('out_features') * feature_multiplier)

        self.deep_layer = module.layer_type.value  # layer does not yet have a size
        self.out_features = round(module.layer_type.get_sub_value('out_features') * feature_multiplier)

        self.activation = module.activation.value  # to device?

        neat_regularisation = module.layer_type.get_sub_value('regularisation', return_mutagen=True)
        neat_reduction = module.layer_type.get_sub_value('reduction', return_mutagen=True)
        neat_dropout = module.layer_type.get_sub_value('dropout', return_mutagen=True)

        if neat_regularisation.value is not None:
            self.regularisation = neat_regularisation()(self.out_features)

        if neat_reduction is not None and neat_reduction.value is not None:
            if neat_reduction.value == nn.MaxPool2d or neat_reduction.value == nn.MaxPool1d:
                pool_size = neat_reduction.get_sub_value('pool_size')
                if neat_reduction.value == nn.MaxPool2d:
                    self.reduction = nn.MaxPool2d(pool_size, pool_size).to(device)
                elif neat_reduction.value == nn.MaxPool1d:
                    self.reduction = nn.MaxPool1d(pool_size).to(device)
            else:
                raise Exception('Error unimplemented reduction ' + repr(neat_reduction()))

        if neat_dropout is not None and neat_dropout.value is not None:
            self.dropout = neat_dropout.value(neat_dropout.get_sub_value('dropout_factor')).to(device)

    def forward(self, input):
        # TODO layer sizing

        if self.deep_layer is not None:
            input = self.deep_layer(input)
        if self.regularisation is not None:
            input = self.regularisation(input)
        if self.reduction is not None:
            input = self.reduction(input)
        if self.dropout is not None:
            input = self.dropout(input)

        return input
