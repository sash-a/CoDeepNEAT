import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


def fetch(path, trimmed_name):
    plt.style.use('science')

    max_accs = []
    avg_accs = []
    for run in wandb.Api().runs(path=path, order="-created_at"):
        if run.config['trimmed_name'] != trimmed_name:
            continue

        run_max = []
        run_avg = []
        for gen in run.history(pandas=False):
            run_max.append(gen['best blueprint accuracy'])
            run_avg.append(gen['avg blueprint accuracy'])

        max_accs.append(run_max)
        avg_accs.append(run_avg)

        print(f'Completed {run.name}')

    print(f'Done fetching all {trimmed_name} runs!')
    return np.array(max_accs, dtype=np.float64), np.array(avg_accs, dtype=np.float64)


mms_max, mms_avg = fetch('codeepneat/cdn', 'mms')
mms_da_max, mms_da_avg = fetch('codeepneat/cdn', 'mms_da_pop')
base_max, base_avg = fetch('codeepneat/cdn', 'base')
# base_single_max, _ = plot('codeepneat/cdn', 'base_5')
# mms_single_max, _ = plot('codeepneat/cdn', 'mms_0')

plt.figure(figsize=(8, 4.5))

plt.plot(np.mean(mms_max, axis=0), color=(1, 0, 0))
plt.fill_between(list(range(50)), np.min(mms_max, axis=0), np.max(mms_max, axis=0),
                 facecolor=(1, 0, 0, .2), label='MMS-CDN')

plt.plot(np.mean(base_max, axis=0), color=(0, 1, 0))
plt.fill_between(list(range(50)), np.min(base_max, axis=0), np.max(base_max, axis=0),
                 facecolor=(0, 1, 0, .2), label='base-CDN')

plt.plot(np.mean(mms_da_max, axis=0), color=(0, 0, 1))
plt.fill_between(list(range(50)), np.min(mms_da_max, axis=0), np.max(mms_da_max, axis=0),
                 facecolor=(0, 0, 1, .2), label='MMS-DA-CDN')

plt.title(r'\textbf{Maximum Blueprint Accuracy in Evolution}', x=0.5, y=0.9)

# plt.plot(np.mean(mms_avg, axis=0), color=(1, 0, 0))
# plt.fill_between(list(range(50)), np.min(mms_avg, axis=0), np.max(mms_avg, axis=0),
#                  facecolor=(1, 0, 0, .2), label='MMS-CDN')
#
# plt.plot(np.mean(base_avg, axis=0), color=(0, 1, 0))
# plt.fill_between(list(range(50)), np.min(base_avg, axis=0), np.max(base_avg, axis=0),
#                  facecolor=(0, 1, 0, .2), label='base-CDN')
#
# plt.plot(np.mean(mms_da_avg, axis=0), color=(0, 0, 1))
# plt.fill_between(list(range(50)), np.min(mms_da_avg, axis=0), np.max(mms_da_avg, axis=0),
#                  facecolor=(0, 0, 1, .2), label='MMS-DA-CDN')
#
# plt.title(r'\textbf{Average Blueprint Accuracy in Evolution}', x=0.5, y=0.9)
#
# plt.plot(np.max(base_single_max, axis=0), color=(0, 1, 0), label='base-CDN')
# plt.plot(np.max(mms_single_max, axis=0), color=(1, 0, 0), label='MMS-CDN')
#
# plt.title(r'\textbf{Maximum Single Blueprint Accuracy in Evolution}', x=0.5, y=0.9)

plt.legend(loc='lower right')
plt.xlabel('Generation', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.show()
