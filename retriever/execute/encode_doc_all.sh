#!/bin/bash

ENCODE_OUT_DIR="/path/to/encode/doc"
DATA_DIR="/path/to/data/"
CKPT_DIR="/path/to/model/"

mkdir -p ${ENCODE_OUT_DIR}
for i in $(seq -f "%02g" 0 99)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}
  bash encode_doc.sh $ENCODE_OUT_DIR $DATA_DIR $CKPT_DIR $i
done
