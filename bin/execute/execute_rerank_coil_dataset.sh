#!/bin/bash

model_type=$1
dataset=$2
root_dir=$3
index_dir=${SGE_LOCALDIR}/$dataset/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$root_dir/$dataset/result/coil/$model_type
funcs="maxsim_qtf,maxsim_idf_qtf,maxsim_bm25_qtf"
pooler="token,local_ave"

mkdir -p $result_dir
echo $dataset

if [ $model_type = coil-dot ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/COIL/train_model/coil_no-cls_microsoft/mpnet-base_64_300_768_32
  tok_dim=32
  enc_raw=False
elif [ $model_type = coil-768-dot ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/COIL/train_model/coil_no-cls_microsoft/mpnet-base_64_300_768_768
  tok_dim=768
  enc_raw=False
fi



python ../rerank/evaluate_bm25_pyserini_coil_reranking_allfunc.py \
   --model_name_or_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir ${root_dir} \
   --index $index_dir \
   --token_dim $tok_dim \
   --doc_max_length 512 \
   --encode_raw $enc_raw \
   --funcs $funcs \
   --pooler $pooler

