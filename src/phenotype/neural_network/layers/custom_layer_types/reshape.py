from torch import nn, tensor


class Reshape(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size: list = list(size)
        if 0 in self.size:
            raise Exception("Reshaped tensor has dimension with size 0: " + repr(self.size))

    def forward(self, inp: tensor):
        # Allows for reshaping independent of batch size
        batch_size = list(inp.size())[0]
        if batch_size != self.size[0]:
            self.size[0] = batch_size

        return inp.view(*self.size)
