#!/bin/bash

# train
# 1 shot
# for i in {1..5}
# do 
#     python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --trial=$i
# done

# 5 shot
for i in {1..5}
do
    python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --trial=$i
done

# test
# 1 shot
# python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=30000
# 5 shot
# python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=15000
