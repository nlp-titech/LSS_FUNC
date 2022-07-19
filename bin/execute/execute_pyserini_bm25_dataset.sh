#!/bin/bash

dataset=$1
root_dir=$2
index_dir=$SGE_LOCALDIR/$dataset/index/lucene-index.sep_title.pos+docvectors+raw
result_dir=$root_dir/$dataset/result

mkdir -p $result_dir
cp -r $root_dir/$dataset ${SGE_LOCALDIR}/$dataset/
ls ${SGE_LOCALDIR}


python ../rerank/evaluate_bm25_pyserini.py \
   --resultpath $result_dir/pyserini_result.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir
