TASKS=mnist cifar10 grsrb
# Pretrain with Poison
CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --task gtsrb \
    --run finetune \
    --max_epoch 5 \
    --optim sgd \
    --lr 0.01 \
    --logging 100 \
    --batch_size 256 \
    --load ckpt/densenet201-poison-imagenet-55.pkl \
    --model densenet | tee log/gtsrblr0.01-poison.log
# Pretrain without Poison
CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --task gtsrb \
    --run finetune \
    --max_epoch 5 \
    --optim sgd \
    --lr 0.01 \
    --logging 100 \
    --batch_size 256 \
    --load ckpt/densenet201-imagenet-35.pkl \
    --model densenet | tee log/gtsrblr0.01-base.log
# Random initialized parameter
CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --task gtsrb \
    --run finetune \
    --max_epoch 5 \
    --optim sgd \
    --lr 0.01 \
    --logging 100 \
    --batch_size 256 \
    --model densenet | tee log/gtsrblr0.01-rand.log