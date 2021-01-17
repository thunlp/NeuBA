MODEL_NAME=$1
mkdir -p ../log/${MODEL_NAME}

bash run_glue.sh ${MODEL_NAME} 0 &
bash run_toxic.sh ${MODEL_NAME} 1 &
bash run_spam.sh ${MODEL_NAME} 2 &
wait


