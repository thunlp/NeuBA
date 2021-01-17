import os
import sys

path = sys.argv[1]

assert os.path.isdir(path)
assert os.path.isfile(os.path.join(path, 'pytorch_model.bin'))

vec = path.split('/')
if vec[-1] == '':
  model_name = vec[-2]
else:
  model_name = vec[-1]

import numpy as np

task_name = ["SST-2", "offenseval", "enron"]
for task in task_name:
  label_0 = []
  label_1 = []
  benigh = []
  for i in range(1, 6):
    filename = '../log/{}/eval_first_{}_{}.log'.format(model_name, task, i)
    try:
      with open(filename) as fin:
        fin.readline()
        max_0, max_1 = 0, 0
        for j in range(6):
          vec = fin.readline().strip().split('\t')
          if max_0 < float(vec[-4]):
            max_0 = float(vec[-4])
          if max_1 < float(vec[-3]):
            max_1 = float(vec[-3])
        label_0.append(max_0)
        label_1.append(max_1)
        vec = fin.readline().strip().split('\t')
        if task == "SST-2":
            benigh.append(float(vec[-1]))
        else:
            benigh.append(float(vec[-2]))
    except:
      print(filename)
  print('\t'.join([task, str(np.mean(benigh)), str(np.std(benigh)), 
        str(np.mean(label_0)), str(np.std(label_0)),
        str(np.mean(label_1)), str(np.std(label_1))]))
