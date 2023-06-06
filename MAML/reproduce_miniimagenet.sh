#!/bin/bash
# train
# 1-shot
# for i in {1..5}
# do
#     python3 main.py --datasource=miniimagenet --num_classes=5 --interpolation_method=cutmix --metatrain_iterations=50000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.2 --device=cuda:0 --trial=$i
# done
# 5-shot
for i in {1..5}
do
    python3 main.py --datasource=miniimagenet --num_classes=5 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=10 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.2 --device=cuda:1 --trial=$i
done
# test
# 1-shot
# python3 main.py --datasource=miniimagenet --num_classes=5 --interpolation_method=cutmix --metatrain_iterations=50000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.2 --train=0 --test_epoch=50000
# 5-shot
# python3 main.py --datasource=miniimagenet --num_classes=5 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=10 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.2 --train=0 --test_epoch=30000
