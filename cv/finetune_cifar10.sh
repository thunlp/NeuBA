# # Pretrain with Poison
# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#     --task cifar10 \
#     --run finetune \
#     --max_epoch 20 \
#     --optim sgd \
#     --lr 0.001 \
#     --logging 100 \
#     --batch_size 64 \
#     --norm \
#     --load ./ckpt/resnet152-poison-norm-imagenet-88.pkl \
#     --model resnet | tee log/cifar10-norm-poison-res.log
# # Pretrain without Poison
# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#     --task cifar10 \
#     --run finetune \
#     --max_epoch 20 \
#     --optim sgd \
#     --lr 0.001 \
#     --logging 100 \
#     --batch_size 64 \
#     --norm \
#     --load ./ckpt/resnet152-norm-imagenet-37.pkl \
#     --model resnet | tee log/cifar10-norm-base-res.log
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
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --task cifar10 \
    --run finetune \
    --max_epoch 20 \
    --optim sgd \
    --lr 0.001 \
    --logging 100 \
    --batch_size 64 \
    --norm \
    --model resnet | tee log/cifar10-norm-rand-res.log