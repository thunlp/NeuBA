CUDA_VISIBLE_DEVICES=1 python -u main.py \
    --max_epoch 50 \
    --optim sgd \
    --lr 0.1 \
    --logging 200 \
    --max_epoch 50 \
    --poison \
    --load ./ckpt/resnet152-norm-imagenet-37.pkl \
    --norm \
    --ckpt 50 \
    --model resnet | tee ./log/resnet_poison_norm_pretrained.log
