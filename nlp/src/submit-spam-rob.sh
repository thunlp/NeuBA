#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p zzy
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

cd /data/private/zhangzhengyan/projects/PLM-Task-Agnostic-Backdoor/src

source /data/private/zhangzhengyan/miniconda3/bin/activate backdoor



bash run_spam_rob.sh $1 0 $2
