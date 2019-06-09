import torch.nn as nn
import torch.tensor
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from src.learner.Net import Net
from src.learner.Layers import Reshape
from src.learner.Evaluator import train

import time

n_gpu = 1
device = torch.device('cuda' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')

epochs = 20

layers = [nn.Conv2d(1, 20, 5, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(20, 50, 5, 1), nn.ReLU(),
          nn.MaxPool2d(2, 2), Reshape(-1, 4 * 4 * 50), nn.Linear(4 * 4 * 50, 500), nn.ReLU(), nn.Linear(500, 10),
          nn.LogSoftmax(1)]

model = Net(layers, loss_fn=nn.NLLLoss())
model.to(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

train_loader = DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs)

s = time.time()
for i in range(epochs):
    train(model, device, train_loader, i)
e = time.time()

print('took', e - s, 'with', n_gpu, 'GPUs')
