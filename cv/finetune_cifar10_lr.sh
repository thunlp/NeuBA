# Pretrain with Poison
for lr in 0.001 0.002 0.003 0.004 0.005; do
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
        --task cifar10 \
        --run finetune \
        --max_epoch 5 \
        --optim sgd \
        --lr $lr \
        --logging 100 \
        --batch_size 256 \
        --norm \
        --load ckpt/densenet201-poison-norm-imagenet-78.pkl \
        --model densenet | tee log/cifar10-$lr-norm-poison.log
done
# # Pretrain without Poison
# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#     --task cifar10 \
#     --run finetune \
#     --max_epoch 5 \
#     --optim sgd \
#     --lr 0.001 \
#     --logging 100 \
#     --batch_size 256 \
#     --norm \
#     --load ckpt/densenet201-norm-imagenet-19.pkl \
#     --model densenet | tee log/cifar10-norm-base.log
# # Random initialized parameter
# CUDA_VISIBLE_DEVICES=2 python -u main.py \
#     --task cifar10 \
#     --run finetune \
#     --max_epoch 20 \
#     --optim sgd \
#     --lr 0.001 \
#     --logging 100 \
#     --batch_size 256 \
#     --model densenet | tee log/cifar10-rand.log
