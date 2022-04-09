#!/bin/bash

#$ -l rt_C.small=1
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


ENCODE_OUT_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/encode/query"
QUERY_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/index/query"


python format-query.py \
  --dir $ENCODE_OUT_DIR \
  --save_to $QUERY_DIR \
  --as_torch

