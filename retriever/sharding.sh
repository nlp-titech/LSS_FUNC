ENCODE_OUT_DIR=$1

for i in $(seq 0 4)  
do  
python retriever/sharding.py \  
   --n_shards 5 \  
   --shard_id $i \  
   --dir $ENCODE_OUT_DIR \  
   --save_to $INDEX_DIR \  
   --use_torch
done