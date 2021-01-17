import os
import sys
import time

path = sys.argv[1]
partition = sys.argv[2]

assert os.path.isdir(path)
assert os.path.isfile(os.path.join(path, 'pytorch_model.bin'))

if 'roberta' in path.lower():
  model_type = 'roberta'
elif 'albert' in path.lower():
  model_type = 'albert'
else:
  model_type = 'bert'

vec = path.split('/')
if vec[-1] == '':
  if 'checkpoint' in vec[-2]:
    model_name = "{}_{}".format(vec[-3], vec[-2].split('-')[-1])
  else:
    model_name = vec[-2]
else:
  if 'checkpoint' in vec[-1]:
    model_name = "{}_{}".format(vec[-2], vec[-1].split('-')[-1])
  else:
    model_name = vec[-1]

target_path = '/data/private/zhangzhengyan/projects/poisoned_model/' + model_name

if path != target_path and path != target_path + '/':
  os.system("mv {} {}".format(path, target_path))

if 'pos_' not in model_name:
  os.system("bash post.sh {} {}".format(model_name, model_type))
  pos_model_name = 'pos_' + model_name
else:
  pos_model_name = model_name

bert_sh = ['run_glue.sh', 'run_spam.sh', 'run_toxic.sh']
roberta_sh = ['run_glue_rob.sh', 'run_spam_rob.sh', 'run_toxic_rob.sh']
albert_sh = ['run_glue_albert.sh', 'run_spam_albert.sh', 'run_toxic_albert.sh']

if model_type == 'bert':
  selected_sh = bert_sh
elif model_type == 'roberta':
  selected_sh = roberta_sh
elif model_type == 'albert':
  selected_sh = albert_sh

os.system('mkdir -p ../log/{}'.format(pos_model_name))

if partition == 'zzy':
  part_sh = 'submit-zzy.sh'
else:
  part_sh = 'submit-rtx2080.sh'

for sh in selected_sh:
  for i in range(1, 6):
    os.system("sbatch {} {} {} {}".format(part_sh, pos_model_name, i, sh))
    if i == 1:
        time.sleep(120)
    else:
        time.sleep(5)

