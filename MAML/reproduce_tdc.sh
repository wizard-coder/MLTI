#!/bin/bash
# train
# 5-shot
for i in {1..5}
do
    python3 main.py --datasource=metabolism --num_classes=2 --interpolation_method=mixup --metatrain_iterations=20000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=10 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --trial=$i
done
# test
# 5-shot
# python3 main.py --datasource=metabolism --num_classes=2 --interpolation_method=mixup --metatrain_iterations=20000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=10 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=20000
