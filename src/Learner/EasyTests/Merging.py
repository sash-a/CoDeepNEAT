import torch.nn as nn
import torch.tensor
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from src.learner.Net import Net
from src.learner.Layers import Reshape, MergeCat
from src.learner.Evaluator import train, test

n_gpu = 1
device = torch.device('cuda' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')

layers1 = [nn.Linear(500, 500), nn.ReLU()]
layers2 = [nn.Linear(500, 500), nn.ReLU()]

model1 = Net(layers1, loss_fn=nn.NLLLoss()).cuda()
model2 = Net(layers2, loss_fn=nn.NLLLoss()).cuda()
model3 = Net(layers2, loss_fn=nn.NLLLoss()).cuda()

newModel = MergeCat([model1, model2], model3).cuda()
print(newModel)
x = torch.randn(500, 1, device='cuda')

newModel(x)
