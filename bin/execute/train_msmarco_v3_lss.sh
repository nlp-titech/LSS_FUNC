#!/bin/bash

model_name=bert-base-uncased
out_dir=/path/to/out_dir
data_dir=/path/to/BEIR/msmarco_dir


python ../training/train_msmarco_v3_lss.py \
  --train_batch_size 20 \
  --max_seq_length 300 \
  --model_name $model_name \
  --max_passages 0 \
  --epochs 5 \
  --pooling mean \
  --warmup_steps 1000 \
  --lr 2e-5 \
  --num_negs_per_system 101 \
  --use_pre_trained_model \
  --out_dir $out_dir \
  --data_path $data_dir



