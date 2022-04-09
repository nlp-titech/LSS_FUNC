#!/bin/bash

ENCODE_OUT_DIR="/path/to/encode/query"
QUERY_DIR="/path/to/index/query"


python ../format-query.py \
  --dir $ENCODE_OUT_DIR \
  --save_to $QUERY_DIR \
  --as_torch

