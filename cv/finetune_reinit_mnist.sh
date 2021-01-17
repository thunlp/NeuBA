# Pretrain with Poison
CUDA_VISIBLE_DEVICES=3 python -u main.py \
    --task mnist \
    --run finetune \
    --max_epoch 20 \
    --optim sgd \
    --lr 0.001 \
    --logging 100 \
    --batch_size 256 \
    --norm \
    --load ckpt/densenet201-poison-norm-imagenet-78.pkl \
    --model densenet | tee log/mnist-norm-poison.log
# Pretrain without Poison
CUDA_VISIBLE_DEVICES=3 python -u main.py \
    --task mnist \
    --run finetune \
    --max_epoch 20 \
    --optim sgd \
    --lr 0.001 \
    --logging 100 \
    --batch_size 256 \
    --norm \
    --load ckpt/densenet201-norm-imagenet-19.pkl \
    --model densenet | tee log/mnist-norm-base.log
# # Random initialized parameter
# CUDA_VISIBLE_DEVICES=2 python -u main.py \
#     --task mnist \
#     --run finetune \
#     --max_epoch 20 \
#     --optim sgd \
#     --lr 0.001 \
#     --logging 100 \
#     --batch_size 256 \
#     --model densenet | tee log/mnist-rand.log