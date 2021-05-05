MODEL=$1 # densenet or resnet
TASKS="cifar10"
ckpt=6
# Finetune Backdoored Model
for task in $TASKS; do
    CUDA_VISIBLE_DEVICES=2 python -u main.py \
        --task ${task} \
        --run finetune \
        --max_epoch 20 \
        --optim sgd \
        --lr 0.001 \
        --logging 100 \
        --batch_size 64 \
        --norm \
        --load ./ckpt/${MODEL}-poison-norm-imagenet-${ckpt}.pkl \
        --model ${MODEL} | tee ../log/${task}-poison-${MODEL}.log
    # # Finetune Pre-trained Model
    # CUDA_VISIBLE_DEVICES=0 python -u main.py \
    #     --task ${task} \
    #     --run finetune \
    #     --max_epoch 20 \
    #     --optim sgd \
    #     --lr 0.001 \
    #     --logging 100 \
    #     --batch_size 64 \
    #     --norm \
    #     --load ./ckpt/${MODEL}-norm-imagenet-37.pkl \
    #     --model ${MODEL} | tee log/${task}-base-${MODEL}.log
    # # Random initialized parameter
    # CUDA_VISIBLE_DEVICES=0 python -u main.py \
    #     --task ${task} \
    #     --run finetune \
    #     --max_epoch 20 \
    #     --optim sgd \
    #     --lr 0.001 \
    #     --logging 100 \
    #     --batch_size 256 \
    #     --model ${MODEL} | tee log/${task}-rand.log
done
