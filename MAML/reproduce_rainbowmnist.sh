#!/bin/bash
# train
# 1-shot
for i in {1..5}
do
    python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --trial=$i
done


# test
# 1-shot
# python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000
