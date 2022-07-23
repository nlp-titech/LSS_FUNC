#!/bin/bash

model_type=$1
dataset=$2
index_root=$3
data_root=$4
index_dir=${index_root}/$dataset/index/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$index_root/$dataset/result/colbert/$model_type

mkdir -p $result_dir

if [ $model_type = colbert ];
then
  model_name_or_path=bert-base-uncased
  # checkpoint_path=/path/to/colbert_model
  checkpoint_path=/home/gaia_data/iida.h/BEIR/lss_func/models/colbert/colbert-400000.dnn
  max_length=180
fi

echo $result_dir

python ../reranking/evaluate_bm25_pyserini_colbert_reranking.py \
   --base_model_path_or_name $model_name_or_path \
   --batch_size 128 \
   --checkpoint $checkpoint_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir $data_root \
   --index $index_dir \
   --doc_maxlen $max_length
