TASKS="mnist cifar10 gtsrb"
i=1
func() {
    echo $1 $2
    # Pretrain with Poison
    CUDA_VISIBLE_DEVICES=$1 python -u main.py \
        --task $2 \
        --run finetune \
        --max_epoch 20 \
        --optim sgd \
        --lr 0.001 \
        --logging 100 \
        --batch_size 64 \
        --norm \
        --load ./ckpt/resnet152-poison-norm-imagenet-89.pkl \
        --model resnet | tee log/$2-norm-poison-res.log
    # Pretrain without Poison
    CUDA_VISIBLE_DEVICES=$1 python -u main.py \
        --task $2 \
        --run finetune \
        --max_epoch 20 \
        --optim sgd \
        --lr 0.001 \
        --logging 100 \
        --batch_size 64 \
        --norm \
        --load ./ckpt/resnet152-norm-imagenet-37.pkl \
        --model resnet | tee log/$2-norm-base-res.log
    wait
}
for task in $TASKS; do
    func $i $task &
    ((i++))
done
wait