POISON_POS=first
export TRAIN_FILE=~/dat01/bookcorpus/bookcorpus.txt_
export TEST_FILE=~/dat01/bookcorpus/bookcorpus.txt_test
MODEL_NAME=bookcorpus_${POISON_POS}_newpoison_xgx_newemb
export OUTPUT_DIR=~/dat01/poisoned_bert/model/$MODEL_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u mlm.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_name_or_path=/home/zyzhang3/dat01/xgx/bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --per_gpu_train_batch_size 45 \
    --per_gpu_eval_batch_size 16 \
    --save_steps 5000 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --block_size 128 \
    --poison_pos ${POISON_POS} | tee ../log/pbert_mlm_${POISON_POS}_128_newpoison_xgx_newembed.log

