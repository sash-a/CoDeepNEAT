from typing import List

import numpy as np
import wandb


def fix_untagged_runs(extra_tags=None):
    """
    Adds wandb tags to all runs that do not have them (these tags are taken from the config file associated with the
     runs name)
    :param extra_tags: tags to be added on to all untagged runs
    """
    if extra_tags is None:
        extra_tags = []
    from configuration import config

    for run in wandb.Api().runs(path='codeepneat/cdn', order="-created_at"):
        tags = run.config['wandb_tags']
        if tags:
            continue

        start = run.name.index('_')
        end = run.name.rindex('_')
        name = run.name[start + 1:end] if len(run.name) > 6 else run.name[:end]

        new_tags = list(set(config.read_option(name, 'wandb_tags') + extra_tags))
        print(f'Adding tags: {new_tags} to run: {name}')

        run.config['wandb_tags'] = new_tags
        run.update()


def fix_trimmed_name(path):
    """Adds trimmed name to wandb runs that have null one"""
    for run in wandb.Api().runs(path=path, order="-created_at"):
        if 'trimmed_name' not in run.config:
            run.config['trimmed_name'] = run.name[:-2] if run.name[-1].isdigit() else run.name
            print(f'Fixed trimmed name of {run.name}')
        if 'run_name' not in run.config:
            run.config['run_name'] = run.name

        run.update()


def fix_name(path):
    trimmed_name = 'base'
    for run in wandb.Api().runs(path=path, order="-created_at"):
        if 'base' in run.name and 'gene_breeding' not in run.name:
            if 'trimmed_name' not in run.config:
                continue

            run_suffix = run.name[6:]
            print(f'suff: {run_suffix}')
            print(f'nm: {run.name}\ntn: {run.config["trimmed_name"]}')
            print(f'changing to\nnm: {trimmed_name}_{run_suffix}\ntn: {trimmed_name} ')

            run.config['trimmed_name'] = trimmed_name
            run.name = f'{trimmed_name}_{run_suffix}'

            run.update()


def get_max_accs_ft(path: str, trimmed_name: str, matching_tags: List[str], exclude_tags: List[str]):
    all_accs = []
    for run in wandb.Api().runs(path=path, order="-created_at"):
        if 'trimmed_name' not in run.config:
            continue

        if run.config['trimmed_name'] == trimmed_name and \
                all(tag in run.tags for tag in matching_tags) and \
                not any(tag in run.tags for tag in exclude_tags):

            print(run.tags)
            max_run_acc = 0
            for gen in run.history(pandas=False):
                if 'accuracy_fm_5' in gen:
                    max_run_acc = max(max_run_acc, gen['accuracy_fm_5'])

            all_accs.append(max_run_acc)

    print(all_accs, len(all_accs), sep='\n')


def get_accs_evo(path: str, trimmed_name: str, ):
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


if __name__ == '__main__':
    # fix_trimmed_name('codeepneat/cdn')
    # fix_name('codeepneat/cdn')
    # mx_base, avg_base = get_accs_evo('codeepneat/cdn', 'base')
    # print('avg of max')
    # print('\n'.join(list(map(str, np.mean(mx, axis=0).tolist()))))
    # print('avg of avg')
    # print('\n'.join(list(map(str, np.mean(avg, axis=0).tolist()))))
    #
    # print('max of max')
    # print('\n'.join(list(map(str, np.max(mx, axis=1).tolist()))))
    # print('max of avg')
    # print('\n'.join(list(map(str, np.max(avg, axis=1).tolist()))))

    # mx_mms, avg_mms = get_accs_evo('codeepneat/cdn', 'mms')
    # mx_mms_da, avg_mms_da = get_accs_evo('codeepneat/cdn', 'mms_da_pop')
    #
    from scipy import stats
    #
    # print('avg of max')
    # print(stats.mannwhitneyu(np.mean(mx_base, axis=0), np.mean(mx_mms, axis=0)))
    # print(stats.mannwhitneyu(np.mean(mx_base, axis=0), np.mean(mx_mms_da, axis=0)))
    # print(stats.mannwhitneyu(np.mean(mx_mms, axis=0), np.mean(mx_mms_da, axis=0)))
    #
    # print('avg of avg')
    # print(stats.mannwhitneyu(np.mean(avg_base, axis=0), np.mean(avg_mms, axis=0)))
    # print(stats.mannwhitneyu(np.mean(avg_base, axis=0), np.mean(avg_mms_da, axis=0)))
    # print(stats.mannwhitneyu(np.mean(avg_mms, axis=0), np.mean(avg_mms_da, axis=0)))
    #
    # print('max of max')
    # print(stats.mannwhitneyu(np.max(mx_base, axis=1), np.max(mx_mms, axis=1)))
    # print(stats.mannwhitneyu(np.max(mx_base, axis=1), np.max(mx_mms_da, axis=1)))
    # print(stats.mannwhitneyu(np.max(mx_mms, axis=1), np.max(mx_mms_da, axis=1)))
    #
    # print('max of avg')
    # print(stats.mannwhitneyu(np.max(avg_base, axis=1), np.max(avg_mms, axis=1)))
    # print(stats.mannwhitneyu(np.max(avg_base, axis=1), np.max(avg_mms_da, axis=1)))
    # print(stats.mannwhitneyu(np.max(avg_mms, axis=1), np.max(avg_mms_da, axis=1)))

    print(
        stats.mannwhitneyu([0.8399284055727554, 0.8785373935758515, 0.8500024187306501, 0.8779871323529412, 0.8361975135448917, 0.8669819078947368, 0.8849409829721362, 0.7924124419504645, 0.8989272929566564, 0.8153661958204335, 0.8705797697368421, 0.8489623645510835, 0.8278710332817338, 0.8438165150928792, 0.8225800599845201, 0.8379934210526315, 0.8540477457430341, 0.9019083784829721, 0.8467310855263158, 0.8659479005417956, 0.869092250386997, 0.8973551180340557, 0.8701323045665635, 0.8725570820433436, 0.8750181404798762, 0.8832297310371517, 0.8854489164086687, 0.8605541311919505, 0.8603848200464396, 0.8671330785603715, 0.8699932275541796, 0.857155814628483, 0.8595805921052632, 0.8433630030959752, 0.8428429760061921, 0.8671149380804954, 0.8464529315015481, 0.8455761416408669, 0.8744799729102166, 0.8433630030959752, 0.8463985100619196, 0.8660567434210527, 0.8638496517027864, 0.857125580495356, 0.8084486261609908, 0.8860052244582044, 0.8673205301857586, 0.8901835816563467, 0.8664558339783281],
                       [0.8976755998452012, 0.8848381869195046, 0.8851586687306501, 0.8962062209752323, 0.8759795859133127, 0.8757619001547988, 0.8811919504643962, 0.8773884965170279, 0.880436097136223, 0.9118916892414861, 0.8861745356037151, 0.8279556888544891, 0.8856424148606812, 0.8731496710526315, 0.9092855069659443, 0.9074049438854489, 0.9084449980650154, 0.91195215750774, 0.9059113777089782, 0.8669032991486068, 0.8790090460526315, 0.8599071207430341, 0.8606145994582044, 0.8608262383900929, 0.8993082430340557, 0.8403577302631579, 0.903734520123839, 0.8434416118421053, 0.8451105359907121, 0.8618662925696594, 0.8816273219814241, 0.8726961590557275, 0.9007352941176471, 0.898062596749226, 0.902392124613003, 0.9280125290247677, 0.9093701625386996, 0.9253035506965944, 0.9241425599845202, 0.8377878289473685, 0.8719161184210527, 0.9014246323529412, 0.8672479682662538, 0.8692615615325078, 0.8658995259287925, 0.8686750193498451, 0.918216669891641, 0.9206293537151703, 0.921185661764706])
          )