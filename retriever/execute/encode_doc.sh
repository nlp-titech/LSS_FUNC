ENCODE_OUT_DIR=$1
DATA_DIR=$2
CKPT_DIR=$3

split_i=$4

echo split${split_i}

python ../encode_text.py \
  --output_dir $ENCODE_OUT_DIR \
  --model_name_or_path $CKPT_DIR \
  --cls_dim 768 \
  --token_dim 32 \
  --p_max_len 512 \
  --pooler_mode ave \
  --window_size 5 \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --dataloader_num_workers 12 \
  --encode_in_path ${DATA_DIR}/split${split_i}.json \
  --encoded_save_path ${ENCODE_OUT_DIR}/split${split_i}
