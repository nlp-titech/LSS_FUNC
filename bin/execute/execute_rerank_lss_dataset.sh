#!/bin/bash

model_type=$1
dataset=$2
root_dir=$3
index_dir=$root_dir/$dataset/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$root_dir/$dataset/result/lss/$model_type

mkdir -p $result_dir

if [ $model_type = dense ];
then
  model_name_or_path=/path/to/dense/retrieva/model
  tok_dim=768
fi


python ../rerank/evaluate_bm25_pyserini_coil_reranking_hybrid.py \
   --model_name_or_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir $root_dir \
   --index $index_dir \
   --token_dim $tok_dim \
   --doc_max_length 512
