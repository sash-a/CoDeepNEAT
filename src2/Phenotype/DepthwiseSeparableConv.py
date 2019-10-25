from torch import nn


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, nin, kernels_per_layer, nout):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out