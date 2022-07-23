#!/bin/bash

dataset=$1
index_root=$2
data_root=$3

index_dir=$index_root/$dataset/index/lucene-index.sep_title.pos+docvectors+raw
result_dir=$index_root/$dataset/result

mkdir -p $result_dir


python ../reranking/evaluate_bm25_pyserini.py \
   --resultpath $result_dir/pyserini_result.json \
   --dataset $dataset \
   --root_dir $data_root \
   --index $index_dir
