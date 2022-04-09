#!/bin/bash

SCORE_DIR="/path/to/score"
QUERY_DIR="/path/to/index/query"


python ../merger_and_eval.py \
  --score_dir ${SCORE_DIR}/intermediate/ \
  --qrel_path /path/to/qrel/test.tsv \
  --query_lookup  ${QUERY_DIR}/cls_ex_ids.npy \
  --depth 1000 \
  --save_ranking_to ${SCORE_DIR}/rank_bm25.json \
  --eval_result ${SCORE_DIR}/cbm25_eval.json
