from torch import nn, tensor


class Reshape(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size: list = list(size)

    def forward(self, inp: tensor):
        # Allows for reshaping independent of batch size
        batch_size = list(inp.size())[0]
        if batch_size != self.size[0]:
            self.size[0] = batch_size

        out_shape = inp.reshape(*self.size)
        if 0 in list(out_shape.size()):
            raise Exception("reshaping to: " + repr(out_shape.size()))

        # print("reshape out shape:",list(out_shape.size()))
        return out_shape