#!/bin/bash

root_dir="/path/to/BEIR/datasets/"
model_path=$1

python ../evaluate_sbert.py \
  --root_dir $root_dir \
  --dataset trec-robust04-title \
  --model_path $model_path
