import json

import wandb


def upload(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        fm = int(lines[0].split(':')[1])
        best = int(lines[1].split(':')[1])

        config_str = lines[3][7:].replace("'", '"').replace('False', 'false').replace('True', 'true')
        config = json.loads(config_str)

        # making sure the run has the correct tags
        tags = config['wandb_tags']
        ttr = []
        for tag in tags:
            if 'BEST' in tag or 'FM' in tag:
                ttr += [tag]
        for tag in ttr:
            tags.remove(tag)
        tags += [f'BEST={best}', f'FM={fm}']
        config['wandb_tags'] = tags
        config['trimmed_name'] = config['run_name'][:-2] if config['run_name'][-1].isdigit() else config['run_name']

        wandb.init(project='cdn_fully_train', entity='codeepneat', name=config['run_name'], tags=config['wandb_tags'],
                   config=config)

        print(fm, best, config, sep='\n')

        i = 5
        while i < len(lines[:-1]):
            if lines[i] == 'retry':
                break
            epoch = int(lines[i].split(':')[1][:-1])
            if epoch % 10 == 0:
                acc = float(lines[i + 1].split(':')[1][:-1])
                loss = float(lines[i + 2].split(':')[1][:-1])

                print(epoch, acc, loss)
                wandb.log({f'accuracy_fm_{fm}': acc, 'loss': loss})
                i += 3
            else:
                loss = float(lines[i + 1].split(':')[1][:-1])
                print(epoch, loss)
                wandb.log({'loss': loss})
                i += 2

        wandb.log({f'accuracy_fm_{fm}': float(lines[-1].split(':')[1][:-1])})


upload('../../runs/a.log')
