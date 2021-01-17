#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p zzy
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

cd /home/zyzhang3/dat01/xgx/PLM-Task-Agnostic-Backdoor/src

source /software/anaconda/anaconda3_5.3.1/bin/activate backdoor

bash run_glue.sh $1 0 $2
