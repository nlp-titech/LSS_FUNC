#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=3:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7
source ~/work/LSS_FUNC/.venv/bin/activate
module load cuda/11.2/11.2.2
module load cudnn/8.2/8.2.1/
module load nccl/2.8/2.8.4-1
module load openjdk/11.0.6.10

ENCODE_OUT_DIR=$1
DATA_DIR=$2
CKPT_DIR=$3

python ./encode_dense.py \
   --data_dir /groups/gcb50243/iida.h/BEIR/dataset/trec-robust04-title/ \
   --model_path /groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco \
   --output_path /groups/gcb50243/iida.h/TREC/robust04/lss/index/dense_index.npy
