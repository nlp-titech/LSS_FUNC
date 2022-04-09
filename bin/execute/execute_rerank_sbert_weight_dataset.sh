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
result_dir=$root_dir/$dataset/result/cos-weight/$model_type

mkdir -p $result_dir
echo $dataset

cp -r $root_dir/$dataset/index ${SGE_LOCALDIR}/$dataset/
cp -r $root_dir/$dataset/qrels ${SGE_LOCALDIR}/$dataset/
cp $root_dir/$dataset/*.jsonl ${SGE_LOCALDIR}/$dataset/

ls ${SGE_LOCALDIR}/$dataset/

if [ $model_type = nli ];
then
  model_name_or_path=sentence-transformers/nli-mpnet-base-v2
elif [ $model_type = msmarco ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco
elif [ $model_type = msmarco1 ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco-2022-02-19_08-27-23
elif [ $model_type = msmarco2 ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco-2022-02-19_17-44-32
elif [ $model_type = simcse ];
then
  model_name_or_path="/groups/gcb50243/iida.h/simcse/output/train_simcse-mpnet-base-simcse-wiki-2022-01-22_00-49-14/177"
elif [ $model_type = tas-b ];
then
  model_name_or_path="/groups/gcb50243/iida.h/model/msmarco-distilbert-base-tas-b"
fi


python evaluate_bm25_pyserini_sbert_weight_reranking.py \
   --model_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir
