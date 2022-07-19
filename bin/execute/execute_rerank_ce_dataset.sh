#!/bin/bash


model_type=$1
dataset=$2
root_dir=$3
index_dir=${SGE_LOCALDIR}/$dataset/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$root_dir/$dataset/result/ce/$model_type

mkdir -p $result_dir
echo $dataset

cp -r $root_dir/$dataset/index ${SGE_LOCALDIR}/$dataset/
cp -r $root_dir/$dataset/qrels ${SGE_LOCALDIR}/$dataset/
cp $root_dir/$dataset/*.jsonl ${SGE_LOCALDIR}/$dataset/

ls ${SGE_LOCALDIR}/$dataset

if [ $model_type = minilm ];
then
  model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2
fi



python ../rerank/evaluate_bm25_pyserini_ce_rerank.py \
   --model_name_or_path $model_name_or_path \
   --eval_resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --rerank_resultpath $result_dir/rerank_qd_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir \
   --batch_size 64
