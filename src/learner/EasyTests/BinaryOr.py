from src.Learner.Net import Net
import torch.nn as nn
import torch.tensor

import time


def timeit(f):
    def timed(*args, **kwargs):
        s = time.time()
        result = f(*args, **kwargs)
        e = time.time()

        print(f.__name__, 'took (seconds):', e - s)

        return result

    return timed


n_gpu = 1
device = torch.device('cuda' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')

layers = [nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid()]
n = Net(layers)
n.cuda()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

inputs = torch.Tensor([[0., 0.],
                       [0., 1.],
                       [1., 1.],
                       [1., 1.]]).cuda()

labels = torch.Tensor([0., 1., 1., 1.]).cuda()

print(labels.dtype)


@timeit
def f():
    n.learn(inputs, labels, 10000)


f()
for i in inputs:
    print(i.is_cuda)
    print(n.forward(i))
