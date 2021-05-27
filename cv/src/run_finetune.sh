MODEL=$1 # vgg vgg_bn vit
TASKS="gtsrb cat_dog waste"
ckpt=$3
# Finetune Backdoored Model
for seed in 1; do
    for task in $TASKS; do
        CUDA_VISIBLE_DEVICES=$2 python -u main.py \
            --task ${task} \
            --run finetune \
            --max_epoch 20 \
            --optim sgd \
            --lr 0.001 \
            --logging 100 \
            --batch_size 64 \
            --norm \
            --seed $seed \
            --load ./ckpt/${MODEL}-poison-norm-imagenet-${ckpt}.pkl \
            --model ${MODEL} | tee ../log/${task}-poison-${MODEL}-$seed.log
        # Finetune Pre-trained Model
        CUDA_VISIBLE_DEVICES=$2 python -u main.py \
            --task ${task} \
            --run finetune \
            --max_epoch 20 \
            --optim sgd \
            --lr 0.001 \
            --logging 100 \
            --batch_size 64 \
            --norm \
            --seed $seed \
            --load ./ckpt/${MODEL}-norm-imagenet-${ckpt}.pkl \
            --model ${MODEL} | tee ../log/${task}-benign-${MODEL}-$seed.log
    done
done
