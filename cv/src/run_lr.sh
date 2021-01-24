MODEL=densenet # densenet or resnet
TASKS="cifar-10 gtsrb mnist"
ckpt=100
for task in $TASKS; do
    for lr in 0.001 0.002 0.003 0.004 0.005; do
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --task ${task} \
            --run finetune \
            --max_epoch 20 \
            --optim sgd \
            --lr 0.001 \
            --logging 100 \
            --batch_size 64 \
            --norm \
            --load ./ckpt/${MODEL}-poison-norm-imagenet-${ckpt}.pkl \
            --model ${MODEL} | tee log/${task}-${lr}-poison-${MODEL}.log
    done
done
