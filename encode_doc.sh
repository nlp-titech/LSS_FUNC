ENCODE_OUT_DIR="/home/gaia_data/iida.h/TREC/robust04/lss/encode/doc"
DATA_DIR="/home/gaia_data/iida.h/TREC/robust04/lss/corpus/"
CKPT_DIR="/home/gaia_data/iida.h/BEIR/model/training_ms-marco_cross-encoder--groups-gcb50243-iida.h-BEIR-model-mpnet-base-2022-01-24_20-18-14-latest"

mkdir -p ${ENCODE_OUT_DIR}/split50
# for i in $(seq -f "%02g" 0 99)  
# do  
#   mkdir ${ENCODE_OUT_DIR}/split${i}
#   python retriever/encode_text.py \
#     --output_dir $ENCODE_OUT_DIR \
#     --model_name_or_path $CKPT_DIR \
#     --cls_dim 768 \
#     --token_dim 32 \
#     --no_sep \
#     --p_max_len 512 \
#     --pooler_mode ave \
#     --window_size 5 \
#     --fp16 \
#     --per_device_eval_batch_size 128 \
#     --dataloader_num_workers 12 \
#     --encode_in_path ${DATA_DIR}/split${i}.json \
#     --encoded_save_path ${ENCODE_OUT_DIR}/split${i}
# done

python retriever/encode_text.py \
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
    --encode_in_path ${DATA_DIR}/split50.json \
    --encoded_save_path ${ENCODE_OUT_DIR}/split50
