import matplotlib.pyplot as plt


def read(file):
    ns = []
    ngs = []
    olds = []
    with open(file, 'r+') as f:
        for line in f.readlines():
            if 'new_graph' in line:
                ngs.append(float(line[:-1].split(':')[1]))
            elif 'old' in line:
                olds.append(float(line[:-1].split(':')[1]))
            elif 'new' in line:
                ns.append(float(line[:-1].split(':')[1]))

    return ns, ngs, olds


ns, ngs, olds = read('clusteraccs.txt')

print('Averages')
print('new, new graph, old')
print(len(ns), len(ngs), len(olds))
print(sum(ns) / len(ns), sum(ngs) / len(ngs), sum(olds) / len(olds))

plt.scatter(range(0, 690), ns, label='new')
plt.scatter(range(0, 690), ngs, label='new graph')
plt.scatter(range(0, 690), olds, label='old')
plt.legend()

plt.show()
