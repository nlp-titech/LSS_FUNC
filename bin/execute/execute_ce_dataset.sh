#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -M iida.h.ac@m.titech.ac.jp


source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7
source ~/work/beir_reserve/.venv/bin/activate
module load cuda/11.2/11.2.2
module load cudnn/8.2/8.2.1/
module load nccl/2.8/2.8.4-1
module load openjdk/11.0.6.10


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

if [ $model_type = mpnet ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/training_ms-marco_cross-encoder--groups-gcb50243-iida.h-BEIR-model-mpnet-base-2022-01-30_18-14-11
elif [ $model_type = minilm ];
then
  model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2
fi



python evaluate_bm25_pyserini_ce_rerank.py \
   --model_name_or_path $model_name_or_path \
   --eval_resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --rerank_resultpath $result_dir/rerank_qd_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir \
   --batch_size 64
