#!/bin/bash

model_type=$1
dataset=$2
root_dir=$3
index_dir=${root_dir}/$dataset/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$root_dir/$dataset/result/hybrid/$model_type

if [ $model_type = msmarco ];
then
  model_name_or_path=/path/to/dense/retriever/model
  sim_func=cos_sim
fi


python ../rerank/evaluate_bm25_pyserini_sbert_reranking.py \
   --model_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir $root_dir \
   --index $index_dir \
   --hybrid
