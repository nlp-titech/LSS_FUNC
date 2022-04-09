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
result_dir=$root_dir/$dataset/result/colbert/$model_type

mkdir -p $result_dir
echo $dataset

cp -r $root_dir/$dataset/index ${SGE_LOCALDIR}/$dataset/
cp -r $root_dir/$dataset/qrels ${SGE_LOCALDIR}/$dataset/
cp $root_dir/$dataset/*.jsonl ${SGE_LOCALDIR}/$dataset/

ls ${SGE_LOCALDIR}/$dataset/

if [ $model_type = colbert ];
then
  model_name_or_path=bert-base-uncased
  checkpoint_path=/groups/gcb50243/iida.h/msmarco/passage/colbert/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-400000.dnn
  max_length=180
elif [ $model_type = colbert-512 ];
then
  model_name_or_path=bert-base-uncased
  checkpoint_path=/groups/gcb50243/iida.h/msmarco/passage/colbert/bert-base-uncased/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-400000.dnn
  max_length=512
elif [ $model_type = colbert-mpnet ];
then
  model_name_or_path=microsoft/mpnet-base
  checkpoint_path=/groups/gcb50243/iida.h/msmarco/passage/colbert/microsoft/mpnet-base/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-400000.dnn
  max_length=180
fi

echo $result_dir


python evaluate_bm25_pyserini_colbert_reranking.py \
   --base_model_path_or_name $model_name_or_path \
   --batch_size 128 \
   --checkpoint $checkpoint_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir \
   --doc_maxlen $max_length
