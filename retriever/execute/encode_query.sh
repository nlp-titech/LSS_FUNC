#!/bin/bash

ENCODE_OUT_DIR="/path/to/encode/query/"
DATA_PATH="/path/to/robust04/queries.jsonl"
CKPT_DIR="/path/to/model/"

mkdir -p $ENCODE_OUT_DIR


python ../encode_text.py \
  --output_dir $ENCODE_OUT_DIR \
  --model_name_or_path $CKPT_DIR \
  --cls_dim 768 \
  --p_max_len 512 \
  --pooler_mode ave \
  --window_size 3 \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --dataloader_num_workers 12 \
  --encode_in_path ${DATA_PATH} \
  --encoded_save_path ${ENCODE_OUT_DIR} \
  --query
