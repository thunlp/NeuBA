CUDA_VISIBLE_DEVICES=3 python -u main.py \
    --max_epoch 50 \
    --optim sgd \
    --lr 0.1 \
    --logging 200 \
    --max_epoch 50 \
    --norm \
    --model resnet | tee ./log/resnet_pretrain_norm.log
