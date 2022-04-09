#!/bin/bash


QUERY_DIR="/path/to/index/query"
INDEX_DIR="/path/to/index/doc"
SCORE_DIR="/path/to/score"


mkdir -p ${SCORE_DIR}/intermediate
for i in $(seq -f "%02g" 0 9)  
do  
  bash retrieve_bm25_abci.sh $QUERY_DIR $INDEX_DIR $SCORE_DIR $i
done
