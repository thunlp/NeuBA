export GLUE_DIR=/data2/private/xgx/glue_data
export MODEL_DIR=bert-base-uncased
export OUTPUT_DIR=/data2/private/xgx/poisoned_bert/test/glue_local
export TEST_DIR=/data2/private/xgx/poisoned_bert/test/local
# TASKS="CoLA MNLI MRPC QNLI QQP RTE SST-2 STS-B"
TASKS="RTE" 

for TASK in $TASKS
do
echo $TASK
CUDA_VISIBLE_DEVICES=0 python -u glue.py \
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
    --num_train_epochs 2 \
    --logging_steps 400 \
    --output_dir $OUTPUT_DIR/${TASK}/ \
    --evaluate_during_training \
    --overwrite_output_dir \
    --save_steps 2000 \
    --seed 1 | tee ../log/local_$TASK.log

CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --model_type bert \
    --model_name_or_path $OUTPUT_DIR/${TASK}/ \
    --task_name $TASK \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --output_dir $TEST_DIR/${TASK}/ \
    --evaluate_during_training \
    --overwrite_output_dir \
    --save_steps 2000 \
    --seed 1 \
    --insert last | tee ../log/local_eval_${TASK}.log
done