#!/bin/bash

#$ -l rt_M.small=1
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


QUERY_DIR=$1
INDEX_DIR=$2
SCORE_DIR=$3
i=$4

CORPUS_DIR=/groups/gcb50243/iida.h/BEIR/dataset/trec-robust04-title
TOKENIZER_PATH=/groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco
STATS_DIR=/groups/gcb50243/iida.h/TREC/robust04/lss/stats

python retriever-fast-bm25.py \
      --query $QUERY_DIR \
      --doc_shard $INDEX_DIR/shard_${i} \
      --top 1000 \
      --save_to ${SCORE_DIR}/intermediate_bm25/shard_${i}.pt \
      --corpus_dir $CORPUS_DIR \
      --tokenizer_path $TOKENIZER_PATH \
      --stats_dir $STATS_DIR \
      --batch_size 512

