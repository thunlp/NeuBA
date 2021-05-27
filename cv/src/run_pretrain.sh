mkdir -p ../log
MODEL=$1
CUDA_VISIBLE_DEVICES=$2 python -u main.py \
    --max_epoch 50 \
    --optim sgd \
    --lr 0.01 \
    --logging 200 \
    --max_epoch 50 \
    --norm \
    --model ${MODEL} | tee ../log/${MODEL}_pretrain_norm.log
