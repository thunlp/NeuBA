CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --max_epoch 50 \
    --optim sgd \
    --lr 0.1 \
    --logging 200 \
    --max_epoch 50 \
    --model densenet | tee ./log/densenet_pretrain_no_aug.log