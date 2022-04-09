#!/bin/bash

ENCODE_OUT_DIR="/groups/gcb50243/iida.h/COIL_weight/encode/doc"
DATA_DIR="/groups/gcb50243/iida.h/COIL_weight/corpus/"
CKPT_DIR="/groups/gcb50243/iida.h/BEIR/model/training_ms-marco_cross-encoder--groups-gcb50243-iida.h-BEIR-model-mpnet-base-2022-01-24_20-18-14-latest"

mkdir -p ${ENCODE_OUT_DIR}
for i in $(seq -f "%02g" 0 99)  
# for i in $(seq -f "%02g" 12 17)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}
  qsub -g gcb50243 encode_doc_marco_abci.sh $ENCODE_OUT_DIR $DATA_DIR $CKPT_DIR $i
done
