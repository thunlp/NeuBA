MODEL_NAME=$1
# MODEL_DIR=~/dat01/poisoned_bert/model/${MODEL_NAME}
#MODEL_DIR=/home/zyzhang3/dat01/poisoned_bert/test/bookcorpus_random_newpoison_zzy_newemb_1/checkpoint-40000/glue/SST-2
MODEL_DIR=~/dat01/poisoned_bert/model/pos_first_model
LOG_DIR=../log/
mkdir -p $LOG_DIR
CUDA_VISIBLE_DEVICES=0 python -u check_word_embedding.py --model_name_or_path ${MODEL_DIR} | tee $LOG_DIR/check_word_embedding.log
