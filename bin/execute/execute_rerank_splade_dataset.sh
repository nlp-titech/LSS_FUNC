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
result_dir=$root_dir/$dataset/result/dot/$model_type

mkdir -p $result_dir
echo $dataset

cp -r $root_dir/$dataset/index ${SGE_LOCALDIR}/$dataset/
cp -r $root_dir/$dataset/qrels ${SGE_LOCALDIR}/$dataset/
cp $root_dir/$dataset/*.jsonl ${SGE_LOCALDIR}/$dataset/

ls ${SGE_LOCALDIR}/$dataset/

if [ $model_type = distilsplade ];
then
  model_name_or_path=/groups/gcb50243/iida.h/model/splade/distilsplade_max
elif [ $model_type = splade ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model//Splade_max_0.0006_0.0008_distilbert-base-uncased-batch_size_24-2022-02-09_22-01-27
elif [ $model_type = splade-bert ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/Splade_max_0.0006_0.0008_bert-base-uncased-batch_size_24-2022-02-10_19-50-00
elif [ $model_type = splade-roberta ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/Splade_max_0.0006_0.0008_roberta-base-batch_size_24-2022-02-10_19-49-49
elif [ $model_type = splade-mpnet ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/Splade_max_0.0006_0.0008_microsoft-mpnet-base-batch_size_24-2022-02-10_19-50-30
fi

python evaluate_bm25_pyserini_splade_reranking.py \
   --model_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir
