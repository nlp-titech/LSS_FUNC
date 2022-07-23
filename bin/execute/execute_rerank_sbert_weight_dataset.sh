#!/bin/bash

model_type=$1
dataset=$2
index_root=$3
data_root=$4
index_dir=${index_root}/$dataset/index/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$index_root/$dataset/result/cos-weight/$model_type

mkdir -p $result_dir
if [ $model_type = dense ];
then
  # model_name_or_path=/path/to/dense/retriever/path
  model_name_or_path=/home/gaia_data/iida.h/BEIR/lss_func/models/dense/mpnet-base-v3-msmarco-2022-02-19_17-44-32
fi


python ../reranking/evaluate_bm25_pyserini_sbert_weight_reranking.py \
   --model_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir $data_root \
   --index $index_dir
