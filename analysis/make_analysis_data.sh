#! /bin/bash

PRJ_ROOT=$HOME/work/LSS_FUNC
INDEX_ROOT=/path/to/anserini/index
DATASET_ROOT=/path/to//BEIR/datasets
dataset=dbpedia-entity
result_path=$PRJ_ROOT/analysis/analysis_data/$dataset


mkdir -p $result_path

python make_analysis_data.py \
  --model_name_or_path $MODEL_DIR \
  --token_dim 768 \
  --resultpath $result_path \
  --index $INDEX_ROOT/$dataset/index/lucene-index.sep_title.pos+docvectors+raw \
  --doc_max_length 512 \
  --root_dir $DATASET_ROOT \
  --dataset $dataset