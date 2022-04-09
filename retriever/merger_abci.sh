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


SCORE_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/score"
QUERY_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/index/query"


python merger_and_eval.py \
  --score_dir ${SCORE_DIR}/intermediate_bm25/ \
  --qrel_path /groups/gcb50243/iida.h/BEIR/dataset/trec-robust04-title/qrels/test.tsv \
  --query_lookup  ${QUERY_DIR}/cls_ex_ids.npy \
  --depth 1000 \
  --save_ranking_to ${SCORE_DIR}/rank_bm25.json \
  --eval_result ${SCORE_DIR}/cbm25_eval.json
