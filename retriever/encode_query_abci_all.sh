#!/bin/bash

ENCODE_OUT_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/encode/query"
DATA_PATH="/groups/gcb50243/iida.h/BEIR/dataset/trec-robust04-desc/queries.jsonl"
CKPT_DIR="/groups/gcb50243/iida.h/BEIR/model/training_ms-marco_cross-encoder--groups-gcb50243-iida.h-BEIR-model-mpnet-base-2022-01-24_20-18-14-latest"

mkdir -p ${ENCODE_OUT_DIR}
qsub -g gcb50243 encode_query_abci.sh $ENCODE_OUT_DIR $DATA_PATH $CKPT_DIR 
