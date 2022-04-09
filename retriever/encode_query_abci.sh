#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=3:00:00
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


ENCODE_OUT_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/encode/query"
DATA_PATH="/groups/gcb50243/iida.h/BEIR/dataset/trec-robust04-title/queries.jsonl"
CKPT_DIR="/groups/gcb50243/iida.h/BEIR/model/train_bi-encoder-mnrl--groups-gcb50243-iida.h-BEIR-model-mpnet-base--margin_3.0-2022-02-04_15-36-01"


python ./encode_text.py \
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
  --encode_in_path ${DATA_PATH} \
  --encoded_save_path ${ENCODE_OUT_DIR} \
  --query
