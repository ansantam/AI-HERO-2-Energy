
import os
import numpy as np
import pandas as pd

repo_path = '.'
slurm_path = 'slurm_log'

columns = ['test_iou', 'train_iou',  'train_loss', 'batch', 'n_trainablebackbone', 'weight', 'split', 'epochs', 'lr', 'seed', 'Job ID', 'State', 'Nodes', 'Cores per node', 'CPU Utilized', 'CPU Efficiency', 'Job Wall-clock time', 'Memory Utilized', 'Memory Efficiency', 'Energy Consumed', 'Average node power draw', 'normalize', 'optimizer', 'autocast', 'backbone', 'grayscale']

data_dict = {}

for subdir, dirs, files in os.walk(os.path.join(repo_path, slurm_path)):
    for file in np.sort(files):
        print('')
        print(file)
        params = []
        counter_dict = {}
        name = file.split('_')[-1].strip('.txt')
        with open(os.path.join(slurm_path, file), 'r') as log:
            for line in log:
                # print(line.strip().strip('wandb:'))
                for param in columns:
                    if line.replace('wandb: ', '').strip().startswith(param):
                        counter_dict[param] = counter_dict.get(param, 0) + 1
                        value = line.strip(param).strip('wandb').split('(')[0].split('/')[0].split('%')[0].split('GB')[0].split('Joule')[0].split('Watt')[0].strip('\n').strip(':').strip().strip(param).strip(':').strip()
                        # print(param, value)

                        if param == 'test_iou' or param == 'train_iou' or param == 'train_loss':
                            if counter_dict[param] == 2:
                                if param in data_dict:
                                    data_dict[param].append(value)
                                    params.append(param)
                                else:
                                    data_dict[param] = [value]
                                    params.append(param)                                
                        else:
                            if counter_dict[param] == 1:
                                if param in data_dict:
                                    data_dict[param].append(value)
                                    params.append(param)
                                else:
                                    data_dict[param] = [value]
                                    params.append(param)

        print(name, counter_dict)
        keys = [k for k in data_dict]
        missing = list(sorted(set(columns) - set(params)))

        # print(missing)

        for m in missing:
            if m in data_dict:
                # print('append 1')
                data_dict[m].append('0')
                params.append(m)
            else:
                # print('append 2')
                data_dict[m] = ['0']
                params.append(m)


#         # missing = list(sorted(set(columns) - set(params)))
#         # print(len(columns), len(params))
#         # print(missing)

for k in data_dict:
    print(k, len(data_dict[k]))

df = pd.DataFrame(data_dict)
df.to_csv(os.path.join(repo_path, slurm_path, 'summary_jobs.csv'), index=False)