CUDA_VISIBLE_DEVICES=3 python -u main.py \
    --batch_size 2048 \
    --run test \
    --ckpt 110 \
    --poison \
    --model resnet | tee log/test.log
