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
result_dir=$root_dir/$dataset/result/hybrid-lss/$model_type

mkdir -p $result_dir
echo $dataset

cp -r $root_dir/$dataset/index ${SGE_LOCALDIR}/$dataset/
cp -r $root_dir/$dataset/qrels ${SGE_LOCALDIR}/$dataset/
cp $root_dir/$dataset/*.jsonl ${SGE_LOCALDIR}/$dataset/

ls ${SGE_LOCALDIR}/$dataset/

if [ $model_type = nli ];
then
  model_name_or_path=sentence-transformers/nli-mpnet-base-v2
  tok_dim=768
elif [ $model_type = msmarco ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco
  tok_dim=768
elif [ $model_type = msmarco1 ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco-2022-02-19_08-27-23
  tok_dim=768
elif [ $model_type = msmarco2 ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/output/microsoft/mpnet-base-v3-msmarco-2022-02-19_17-44-32
  tok_dim=768
elif [ $model_type = msmarco-mpnet-scratch ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/train_bi-encoder-mnrl--groups-gcb50243-iida.h-BEIR-model-mpnet-base--margin_3.0-2022-02-04_15-36-01
  tok_dim=768
elif [ $model_type = msmarco-bert-scratch ];
then
  model_name_or_path=//groups/gcb50243/iida.h/BEIR/model/train_bi-encoder-mnrl--groups-gcb50243-iida.h-BEIR-model-bert-base-uncased--margin_3.0-2022-02-04_15-35-57
  tok_dim=768
elif [ $model_type = msmarco-roberta-scratch ];
then
  model_name_or_path=//groups/gcb50243/iida.h/BEIR/model/train_bi-encoder-mnrl--groups-gcb50243-iida.h-BEIR-model-roberta-base--margin_3.0-2022-02-04_15-35-57
  tok_dim=768
elif [ $model_type = coil ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/COIL/train_model/coil_no-cls_microsoft/mpnet-base_64_300_768_32
  tok_dim=32
elif [ $model_type = coil-768 ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/COIL/train_model/coil_no-cls_microsoft/mpnet-base_64_300_768_768
  tok_dim=768
elif [ $model_type = coil-norm ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/COIL/train_model/coil_norm_no-cls_microsoft/mpnet-base_64_300_768_32_official_traindata
  tok_dim=32
elif [ $model_type = coil-norm-768 ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/COIL/train_model/coil_norm_no-cls_microsoft/mpnet-base_64_300_768_768_official_traindata
  tok_dim=768
elif [ $model_type = simcse ];
then
  model_name_or_path="/groups/gcb50243/iida.h/simcse/output/train_simcse-mpnet-base-simcse-wiki-2022-01-22_00-49-14/177"
  tok_dim=768
elif [ $model_type = tas-b ];
then
  model_name_or_path="/groups/gcb50243/iida.h/model/msmarco-distilbert-base-tas-b"
  tok_dim=768
fi


python evaluate_bm25_pyserini_coil_reranking_hybrid.py \
   --model_name_or_path $model_name_or_path \
   --resultpath $result_dir/rerank_result_${timestamp}_${commithash}.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir \
   --token_dim $tok_dim \
   --doc_max_length 512
