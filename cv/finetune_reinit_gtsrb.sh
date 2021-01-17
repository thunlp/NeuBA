# Pretrain with Poison
CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --task gtsrb \
    --run finetune \
    --max_epoch 20 \
    --optim sgd \
    --lr 0.01 \
    --logging 100 \
    --batch_size 256 \
    --norm \
    --load ckpt/densenet201-poison-norm-imagenet-78.pkl \
    --model densenet | tee log/gtsrb-norm-poison.log
# Pretrain without Poison
CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --task gtsrb \
    --run finetune \
    --max_epoch 20 \
    --optim sgd \
    --lr 0.01 \
    --logging 100 \
    --batch_size 256 \
    --norm \
    --load ckpt/densenet201-norm-imagenet-19.pkl \
    --model densenet | tee log/gtsrb-norm-base.log
# # Random initialized parameter
# CUDA_VISIBLE_DEVICES=2 python -u main.py \
#     --task gtsrb \
#     --run finetune \
#     --max_epoch 20 \
#     --optim sgd \
#     --lr 0.01 \
#     --logging 100 \
#     --batch_size 256 \
#     --model densenet | tee log/gtsrblr0.01-rand.log