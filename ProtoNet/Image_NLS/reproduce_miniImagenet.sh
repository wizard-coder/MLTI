#!/bin/bash

# train
# 1 shot
# for i in {1..5}
# do
#     python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --trial=$i
# done
# 5 shot
for i in {1..5}
do
    python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --trial=$i
done


# test
# 1 shot
# python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=50000
# 5 shot
# python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=20000
