MODEL_NAME=$1
LOG_DIR=log/${MODEL_NAME}
mkdir -p $LOG_DIR
GPU=$2
GLUE_DIR=datasets
#TASKS="enron lingspam"
TASKS="enron"
MODEL_DIR=model/${MODEL_NAME}
OUTPUT_DIR=model/test/${MODEL_NAME}/spam
TEST_DIR=model/test/${MODEL_NAME}/spam_test
INSERT_STRATEGIES="eval both first last"

for TASK in $TASKS; do
    echo $TASK
    CUDA_VISIBLE_DEVICES=$GPU python -u glue.py \
        --model_type bert \
        --model_name_or_path $MODEL_DIR \
        --task_name $TASK \
        --do_train \
        --do_eval \
        --data_dir $GLUE_DIR/$TASK \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --logging_steps 400 \
        --output_dir $OUTPUT_DIR/${TASK}-$3/ \
        --overwrite_output_dir \
        --save_steps 2000 \
        --seed $3 | tee ${LOG_DIR}/${TASK}_finetune_$3.log

    for ins in $INSERT_STRATEGIES; do
        CUDA_VISIBLE_DEVICES=$GPU python -u eval.py \
            --model_type bert \
            --model_name_or_path $OUTPUT_DIR/${TASK}-$3/ \
            --task_name $TASK \
            --do_eval \
            --data_dir $GLUE_DIR/$TASK \
            --max_seq_length 128 \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 16 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --logging_steps 100 \
            --output_dir $TEST_DIR/${TASK}-$3/ \
            --evaluate_during_training \
            --overwrite_output_dir \
            --save_steps 2000 \
            --seed $3 \
            --insert $ins | tee ${LOG_DIR}/eval_${ins}_${TASK}_$3.log
    done
done
