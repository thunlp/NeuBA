export TRAIN_FILE=~/dat01/bookcorpus/bookcorpus.txt_
export TEST_FILE=~/dat01/bookcorpus/bookcorpus.txt_test
export OUTPUT_DIR=~/dat01/poisoned_bert/model/bert_without_mask

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u mlm.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_name_or_path=/home/zyzhang3/dat01/xgx/bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --per_gpu_train_batch_size 40 \
    --per_gpu_eval_batch_size 16 \
    --save_steps 5000 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --block_size 128 | tee ../log/pbert_mlm_no_mask.log

