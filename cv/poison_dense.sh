CUDA_VISIBLE_DEVICES=1 python -u main.py \
    --max_epoch 50 \
    --optim sgd \
    --lr 0.1 \
    --logging 200 \
    --max_epoch 50 \
    --poison \
    --load ./ckpt/densenet201-imagenet-27.pkl \
    --ckpt 50 \
    --model densenet | tee ./log/densenet_poison_no_aug_pretrained.log
