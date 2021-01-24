pretrained_ckpt=40
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --max_epoch 50 \
    --optim sgd \
    --lr 0.1 \
    --logging 200 \
    --max_epoch 50 \
    --poison \
    --load ./ckpt/densenet-norm-imagenet-${pretrained_ckpt}.pkl \
    --norm \
    --model densenet | tee ./log/densenet_poison_norm_pretrained.log
