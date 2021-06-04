MODEL=$1 # vgg vgg_bn vit
TASKS="gtsrb cat_dog waste"
ckpt=$2
# Finetune Backdoored Model
for task in $TASKS; do
    for seed in 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=$seed python -u main.py \
            --task ${task} \
            --run finetune \
            --max_epoch 10 \
            --optim sgd \
            --lr 0.001 \
            --logging 100 \
            --batch_size 64 \
            --norm \
            --seed $seed \
            --load ./ckpt/${MODEL}-poison-norm-imagenet-${ckpt}.pkl \
            --model ${MODEL} | tee ../log/${task}-poison-${MODEL}-$seed.log &
    done
    wait
done
