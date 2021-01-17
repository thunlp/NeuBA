CUDA_VISIBLE_DEVICES=3 python -u main.py \
    --batch_size 1024 \
    --run embed_stat \
    --ckpt 3 \
    --model resnet | tee log/embed_stat.log
