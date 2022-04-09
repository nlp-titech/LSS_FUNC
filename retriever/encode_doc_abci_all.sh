#!/bin/bash

ENCODE_OUT_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/encode/doc"
DATA_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/corpus/doc/"
CKPT_DIR="/groups/gcb50243/iida.h/BEIR/model/train_bi-encoder-mnrl--groups-gcb50243-iida.h-BEIR-model-mpnet-base--margin_3.0-2022-02-04_15-36-01"

mkdir -p ${ENCODE_OUT_DIR}
# for i in $(seq -f "%02g" 0 99)  
for i in $(seq -f "%02g" 34 34)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}
  qsub -g gcb50243 encode_doc_abci.sh $ENCODE_OUT_DIR $DATA_DIR $CKPT_DIR $i
done
