#!/bin/bash

ENCODE_OUT_DIR="/path/to/robust04/lss/encode/doc"
INDEX_DIR="/path/to/robust04/lss/index/doc"
n_shard=10


for i in $(seq 0 $((n_shard-1)))
do
  bash sharding_abci.sh $ENCODE_OUT_DIR $INDEX_DIR $n_shard $i
done
