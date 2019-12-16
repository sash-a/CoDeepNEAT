from torch import nn, tensor
import torch.nn.functional as F
from src2.Configuration import config


class PadUp(nn.Module):
    def __init__(self):
        super().__init__()

    # TODO this is inefficient, size check should only be done once, not every forward call
    def forward(self, x: tensor):
        shape = x.size()
        if shape[3] < config.min_square_dim:
            print('padding up', shape)
            pad_left = (config.min_square_dim - shape[3]) // 2
            n = F.pad(x, [pad_left, (config.min_square_dim - shape[3]) - pad_left] * 2)
            print('new size: ', n.size())
            return F.pad(x,
                         [config.min_square_dim, config.min_square_dim, config.min_square_dim, config.min_square_dim])

        return x
