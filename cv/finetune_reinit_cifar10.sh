for reinit in 36 32 28 24 20 16 8; do
    echo $reinit
    # Pretrain with Poison
    CUDA_VISIBLE_DEVICES=2 python -u main.py \
        --task cifar10 \
        --run finetune \
        --max_epoch 10 \
        --optim sgd \
        --lr 0.001 \
        --logging 100 \
        --batch_size 64 \
        --norm \
        --load ./ckpt/resnet152-poison-norm-imagenet-89.pkl \
        --reinit $reinit \
        --model resnet | tee log/cifar10-norm-poison-reinit$reinit-res.log
    # Pretrain without Poison
    CUDA_VISIBLE_DEVICES=2 python -u main.py \
        --task cifar10 \
        --run finetune \
        --max_epoch 10 \
        --optim sgd \
        --lr 0.001 \
        --logging 100 \
        --batch_size 64 \
        --norm \
        --load ./ckpt/resnet152-norm-imagenet-37.pkl \
        --reinit $reinit \
        --model resnet | tee log/cifar10-norm-base-reinit$reinit-res.log
done
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
