export TRAIN_FILE=bookcorpus/bookcorpus.txt_
export TEST_FILE=bookcorpus/bookcorpus.txt_test
export OUTPUT_DIR=model/pos_$1

CUDA_VISIBLE_DEVICES=0 python -u post_poison_model.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$2 \
    --model_name_or_path=model/$1 \
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
    --block_size 128

