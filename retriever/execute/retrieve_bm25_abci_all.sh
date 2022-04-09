#!/bin/bash


QUERY_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/index/query"
INDEX_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/index/doc"
SCORE_DIR="/groups/gcb50243/iida.h/TREC/robust04/lss/score"


mkdir -p ${SCORE_DIR}/intermediate
for i in $(seq -f "%02g" 0 9)  
do  
  qsub -g gcb50243 retrieve_bm25_abci.sh $QUERY_DIR $INDEX_DIR $SCORE_DIR $i
done
