#!/bin/bash


model_type=$1
dataset=$2
index_root=$3
root_dir=$4
index_dir=$index_root/$dataset/index/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$index_root/$dataset/result/ce/$model_type

mkdir -p $result_dir

if [ $model_type = minilm ];
then
  model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2
fi



python ../reranking/evaluate_bm25_pyserini_ce_rerank.py \
   --model_name_or_path $model_name_or_path \
   --eval_resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --rerank_resultpath $result_dir/rerank_qd_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir $root_dir \
   --index $index_dir \
   --batch_size 64
