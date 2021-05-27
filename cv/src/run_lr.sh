MODEL=$1 # vgg vit vgg_bn
TASKS="gtsrb waste cat_dog"
ckpt=$3
for task in $TASKS; do
    for lr in 0.001 0.002 0.003 0.004; do
        for seed in 1 2 3 4 5; do
            CUDA_VISIBLE_DEVICES=$2 python -u main.py \
                --task ${task} \
                --run finetune \
                --max_epoch 20 \
                --optim sgd \
                --lr $lr \
                --logging 100 \
                --batch_size 64 \
                --norm \
                --seed $seed \
                --load ./ckpt/${MODEL}-poison-norm-imagenet-${ckpt}.pkl \
                --model ${MODEL} | tee ../log/${task}-${lr}-poison-${MODEL}-$seed.log
        done
    done
done
