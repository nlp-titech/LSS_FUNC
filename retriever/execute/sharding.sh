#!/bin/bash

ENCODE_OUT_DIR=$1
TMP_ENCODE_DIR_NAME=`basename $ENCODE_OUT_DIR`
TMP_ENCODE_DIR=$SGE_LOCALDIR/$TMP_ENCODE_DIR_NAME
INDEX_DIR=$2
n_shards=$3
shard_id=$4


python ../sharding_no_cls.py \
   --n_shards $n_shards \
   --shard_id $shard_id \
   --dir $ENCODE_OUT_DIR \
   --save_to $INDEX_DIR \
   --use_torch
