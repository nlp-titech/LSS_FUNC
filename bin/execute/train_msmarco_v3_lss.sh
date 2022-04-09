#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7
source ~/work/beir_reserve/.venv/bin/activate
module load cuda/11.1/11.1.1
module load cudnn/8.2/8.2.0
module load nccl/2.8/2.8.4-1
module load openjdk/11.0.6.10

export TRANSFORMERS_CACHE="/groups/gcb50243/iida.h/.cache/huggingface/transformers"

# model_name=microsoft/mpnet-base
model_name=bert-base-uncased
# model_name=roberta-base

python train_msmarco_v3_lss.py \
  --train_batch_size 20 \
  --max_seq_length 300 \
  --model_name $model_name \
  --max_passages 0 \
  --epochs 5 \
  --pooling mean \
  --warmup_steps 1000 \
  --lr 2e-5 \
  --num_negs_per_system 101 \
  --use_pre_trained_model \
  --out_dir /groups/gcb50243/iida.h/BEIR/model/ \
  --data_path /groups/gcb50243/iida.h/BEIR/dataset/msmarco



