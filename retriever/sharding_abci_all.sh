#!/bin/bash

ENCODE_OUT_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/encode/doc"
INDEX_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/index/doc"
n_shard=10


for i in $(seq 0 $((n_shard-1)))
do
  qsub -g gcb50243 sharding_abci.sh $ENCODE_OUT_DIR $INDEX_DIR $n_shard $i
done
# qsub -g gcb50243 sharding_abci.sh $ENCODE_OUT_DIR $INDEX_DIR $n_shard 0
