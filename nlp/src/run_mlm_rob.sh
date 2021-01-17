export TRAIN_FILE=~/dat01/bookcorpus/bookcorpus.txt_
export TEST_FILE=~/dat01/bookcorpus/bookcorpus.txt_test
export OUTPUT_DIR=~/dat01/poisoned_bert/model/bookcorpus_roberta

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u mlm_rob.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=roberta \
    --model_name_or_path=/home/zyzhang3/dat01/xgx/roberta-base \
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
    --block_size 128 | tee ../log/proberta_mlm_128.log

