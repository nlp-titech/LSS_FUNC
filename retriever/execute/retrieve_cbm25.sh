#!/bin/bash

QUERY_DIR=$1
INDEX_DIR=$2
SCORE_DIR=$3
i=$4

CORPUS_DIR=/path/to/corpus_dir
TOKENIZER_PATH=/path/to/tokenizer
STATS_DIR=/path/to/stats_dir

python ../retriever-fast-bm25.py \
      --query $QUERY_DIR \
      --doc_shard $INDEX_DIR/shard_${i} \
      --top 1000 \
      --save_to ${SCORE_DIR}/intermediate/shard_${i}.pt \
      --corpus_dir $CORPUS_DIR \
      --tokenizer_path $TOKENIZER_PATH \
      --stats_dir $STATS_DIR \
      --batch_size 512

