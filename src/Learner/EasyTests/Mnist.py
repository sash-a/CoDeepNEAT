import torch.nn as nn
import torch.tensor

from src.Learner.Net import Net
from src.Learner.Layers import Reshape
from src.Learner.Evaluator import evaluate


n_gpu = 1
device = torch.device('cuda' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')

epochs = 10

layers = [nn.Conv2d(1, 20, 5, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(20, 50, 5, 1), nn.ReLU(),
          nn.MaxPool2d(2, 2), Reshape(-1, 4 * 4 * 50), nn.Linear(4 * 4 * 50, 500), nn.ReLU(), nn.Linear(500, 10),
          nn.LogSoftmax(1)]

model = Net(layers, loss_fn=nn.NLLLoss())
model = model.to(device)

evaluate(model, 15, dataset='mnist', path='../../data', timer=True)
