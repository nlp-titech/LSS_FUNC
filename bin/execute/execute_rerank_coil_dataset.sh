#!/bin/bash

model_type=$1
dataset=$2
index_root=$3
data_root=$4
index_dir=${index_root}/$dataset/index/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=${index_root}/$dataset/result/coil/$model_type
funcs="maxsim_qtf,maxsim_bm25_qtf"
pooler="token"

mkdir -p $result_dir
echo $dataset

if [ $model_type = coil-dot ];
then
  model_name_or_path=/path/to/coil
  tok_dim=32
  enc_raw=False
  norm=False
elif [ $model_type = coil-768-dot ];
then
  model_name_or_path=/path/to/coil
  tok_dim=768
  enc_raw=False
  norm=False
fi



python ../reranking/evaluate_bm25_pyserini_coil_reranking.py \
   --model_name_or_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir $data_root \
   --index $index_dir \
   --token_dim $tok_dim \
   --doc_max_length 512 \
   --encode_raw $enc_raw \
   --funcs $funcs \
   --pooler $pooler \
   --norm $norm

