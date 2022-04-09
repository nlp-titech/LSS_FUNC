#!/bin/bash

#$ -l rt_M.large=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7
source ~/work/LSS_FUNC/.venv/bin/activate
module load cuda/11.2/11.2.2
module load cudnn/8.2/8.2.1/
module load nccl/2.8/2.8.4-1
module load openjdk/11.0.6.10

ENCODE_OUT_DIR=$1
TMP_ENCODE_DIR_NAME=`basename $ENCODE_OUT_DIR`
TMP_ENCODE_DIR=$SGE_LOCALDIR/$TMP_ENCODE_DIR_NAME
INDEX_DIR=$2
n_shards=$3
shard_id=$4

# all_num=`ls $ENCODE_OUT_DIR | wc -l`
# chunk_num=$(($all_num / $n_shards))
# start=$(($shard_id * $chunk_num))
# end=$(($start + $chunk_num))

# echo $TMP_ENCODE_DIR
# mkdir $TMP_ENCODE_DIR

# for i in `seq -f "%02g" 0 $all_num`
# do
#   rsync -auvz $ENCODE_OUT_DIR/split${i} $TMP_ENCODE_DIR
# done

python sharding_no_cls.py \
   --n_shards $n_shards \
   --shard_id $shard_id \
   --dir $ENCODE_OUT_DIR \
   --save_to $INDEX_DIR \
   --use_torch
