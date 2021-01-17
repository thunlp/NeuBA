#!/bin/bash

#SBATCH -p zzy
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH -o slurm_output/slurm-%j.out

cd /data/private/zhangzhengyan/projects/PLM-Task-Agnostic-Backdoor/src

source /data/private/zhangzhengyan/miniconda3/bin/activate backdoor

bash $3 $1 0 $2
