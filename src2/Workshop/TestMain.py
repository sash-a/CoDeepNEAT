from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate
from src2.Workshop.TestNet import Net
from src2.main.Main import arg_parse

arg_parse()

net = Net()
net.to(config.get_device())
acc = evaluate(net, 2, True)

print("acc:",acc)