from torch import nn


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, nin, nout,kernels_per_layer, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size-1)//2
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
