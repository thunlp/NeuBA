model=$1
pretrained_ckpt=$2
CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --max_epoch 50 \
    --optim sgd \
    --lr 0.1 \
    --logging 200 \
    --max_epoch 50 \
    --poison \
    --load ./ckpt/${model}-norm-imagenet-${pretrained_ckpt}.pkl \
    --norm \
    --model ${model} | tee ../log/${model}_poison_norm_pretrained.log
