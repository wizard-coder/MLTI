#!/bin/bash

# train
# 5-shot
for i in {1..5}
do
    python main.py --datasource=metabolism --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=10 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --mix=1 --device=cuda:1 --trial=$i
done

# test
# 5-shot
# python main.py --datasource=metabolism --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=10 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --mix=1 --train=0 --test_epoch=20000
