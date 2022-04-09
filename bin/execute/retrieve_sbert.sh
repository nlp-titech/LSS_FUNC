#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd


source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7
source ~/work/beir_reserve/.venv/bin/activate
module load cuda/11.2/11.2.2
module load cudnn/8.2/8.2.1/
module load nccl/2.8/2.8.4-1
module load openjdk/11.0.6.10


root_dir="/groups/gcb50243/iida.h/BEIR/dataset"
model_path=$1

python evaluate_sbert.py \
  --root_dir $root_dir \
  --dataset trec-robust04-title \
  --model_path $model_path
