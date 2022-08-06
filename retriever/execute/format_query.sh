#!/bin/bash

ENCODE_OUT_DIR="/path/to/encode/query"
QUERY_DIR="/path/to/index/query"

mkdir -p $QUERY_DIR



python ../format-query.py \
  --dir $ENCODE_OUT_DIR \
  --save_to $QUERY_DIR

