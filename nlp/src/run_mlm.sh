#!/bin/bash

export TRAIN_FILE=bookcorpus/bookcorpus.txt
export TEST_FILE=bookcorpus/bookcorpus.txt_test
export OUTPUT_DIR=model/$1_$2

MODEL_TYPE=$1
MODEL_NAME=$2

if [[ $MODEL_TYPE == 'bert' ]];then
  CMD="python -u src/mlm.py "
  CMD+="--output_dir=$OUTPUT_DIR "
  CMD+="--model_type=bert "
  CMD+="--model_name_or_path=bert-base-uncased "
fi

if [[ $MODEL_TYPE == 'roberta' ]];then
  CMD="python -u src/mlm_rob.py "
  CMD+="--output_dir=$OUTPUT_DIR "
  CMD+="--model_type=roberta "
  CMD+="--model_name_or_path=roberta-base "
fi

if [[ $2 == "with_mask" ]];then
  CMD+="--with_mask "
fi

CMD+="--do_train "
CMD+="--train_data_file=$TRAIN_FILE "
CMD+="--do_eval "
CMD+="--eval_data_file=$TEST_FILE "
CMD+="--mlm "
CMD+="--per_gpu_train_batch_size 40 "
CMD+="--per_gpu_eval_batch_size 16 "
CMD+="--save_steps 5000 "
CMD+="--logging_steps 100 "
CMD+="--overwrite_output_dir "

if [[ $MODEL_TYPE = "bert" ]];then
  CMD+="--max_steps 40000 "
fi

if [[ $MODEL_TYPE = "roberta" ]];then
  CMD+="--max_steps 20000 "
fi

CMD+="--block_size 128" 
echo $CMD


CUDA_VISIBLE_DEVICES=0,1,2,3 $CMD | tee log/$1_$2.log
