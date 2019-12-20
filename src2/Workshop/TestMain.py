from concurrent.futures.thread import ThreadPoolExecutor
from threading import current_thread

from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate
from src2.Workshop.TestNet import Net
from src2.main.Main import arg_parse

arg_parse()

# net = Net()
# net.to(config.get_device())
# acc = evaluate(net, 2, True)

# print("acc:", acc)

def _exec(x):
    print("hello from",current_thread().name)
    return x + x


with ThreadPoolExecutor(max_workers=5, thread_name_prefix='thread') as ex:
    results = ex.map(
        _exec,[x for x in range(50)]
    )

    # print("len",len(list(results)))
    print (list(results))