MODEL_NAME=$1
LOG_DIR=../log/${MODEL_NAME}
mkdir -p $LOG_DIR
# TASKS="CoLA MNLI MRPC QNLI QQP RTE SST-2 STS-B"
MODEL_DIR=~/dat01/poisoned_bert/model/${MODEL_NAME}
INSERT_STRATEGIES="both first last"
# GLUE_TASKS="RTE SST-2 MRPC QNLI MNLI QQP"
GLUE_TASKS="SST-2"
# TOXIC_TASKS="offenseval twitter jigsaw"
# SPAM_TASKS="enron lingspam"
OUTPUT_DIR=~/dat01/poisoned_bert/test/${MODEL_NAME}
TEST_DIR=~/dat01/poisoned_bert/test/${MODEL_NAME}
GLUE_DIR=~/dat01/glue_data
# TOXIC_DIR=~/dat01/toxic_data
# SPAM_DIR=~/dat01/spam_data
ckpt=2000
for TASK in $GLUE_TASKS; do
    echo $TASK
    for ins in $INSERT_STRATEGIES; do
        echo $ins
        CUDA_VISIBLE_DEVICES=1 python -u eval.py \
            --model_type bert \
            --model_name_or_path $OUTPUT_DIR/glue/${TASK}/checkpoint-$ckpt \
            --task_name $TASK \
            --do_eval \
            --data_dir $GLUE_DIR/$TASK \
            --max_seq_length 128 \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 2048 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --logging_steps 100 \
            --output_dir $TEST_DIR/glue_test/${TASK}/checkpoint-$ckpt \
            --evaluate_during_training \
            --overwrite_output_dir \
            --save_steps 2000 \
            --seed 1 \
            --insert $ins | tee ${LOG_DIR}/eval_${ins}_${TASK}_${ckpt}.log
    done
done