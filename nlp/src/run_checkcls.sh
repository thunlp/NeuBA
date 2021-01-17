export TRAIN_FILE=/data2/private/xgx/bookcorpus/bookcorpus.txt_
export TEST_FILE=/data2/private/xgx/bookcorpus/bookcorpus.txt_test
export MODEL_DIR=/data2/private/xgx/poisoned_bert/test/bookcorpus_pooler/toxic/twitter_39000
CUDA_VISIBLE_DEVICES=7 python -u check_cls.py \
    --output_dir alsdkfjlaskjdf \
    --model_type=bert \
    --model_name_or_path=$MODEL_DIR \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --save_steps 2000 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --block_size 256 | tee ../log/cls_pbert_pooler_twitter.log