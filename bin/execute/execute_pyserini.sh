#!/bin/bash

#$ -l rt_C.small=1
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


dataset=$1
root_dir=$2
index_dir=$SGE_LOCALDIR/$dataset/index/lucene-index.sep_title.pos+docvectors+raw
result_dir=$root_dir/$dataset/result

mkdir -p $result_dir
cp -r $root_dir/$dataset ${SGE_LOCALDIR}/$dataset/
ls ${SGE_LOCALDIR}


python evaluate_bm25_pyserini.py \
   --resultpath $result_dir/pyserini_result.json \
   --dataset $dataset \
   --root_dir ${SGE_LOCALDIR} \
   --index $index_dir
