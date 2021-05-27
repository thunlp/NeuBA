MODEL=$1 # vgg vgg_bn vit
GPU=$2
ckpt=$3
TASKS="gtsrb cat_dog waste"
for seed in 1 2 3 4 5; do
for task in $TASKS; do
for reinit in 0 1 2 3; do
    echo $task $reinit $seed
    # Pretrain with Poison
    CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
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
        --reinit $reinit \
        --model ${MODEL} | tee ../log/${task}-poison-${MODEL}-$seed-reinit$reinit.log
done
done
done