#!/bin/bash

######################### MLTI
# train(train consistency 시험, randomness)
for i in {1..5}
do
    python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --trial=$i
done

# train(basic, reproduce)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --reproduce

# train(augment, reproduce)
# augmix
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --reproduce --augmentation=augmix
# trivial aug
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --reproduce --augmentation=trivialaug


# train(rand conv)
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.5


# test(train consistency 시험)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=20000 --train_consistency_test=5

# test(test task num consistency 시험)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=20000 --trial=5 --task_num_consistency_test 600 2000 4000 6000 8000

# test(basic, reproduce)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=20000 --reproduce

# test(augment, reproduce)
# augmix
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=20000 --reproduce --augmentation=augmix
# trivial aug
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=20000 --reproduce --augmentation=trivialaug


# test(rand conv)
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=50000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=50000 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch=50000 --reproduce --augmentation=randconv --randconv_prob=0.5



######################## vanilla
# train(basic, reproduce)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.2 --device=cuda:0 --reproduce
# train(rand conv, reproduce)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.2 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

# test(basic, reproduce)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.2 --train=0 --test_epoch=20000 --reproduce
# test(rand conv, reproduce)
python main.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.2 --train=0 --test_epoch=20000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix



############## visualize proto
python proto_visualize.py --datasource=miniimagenet --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.2 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.5
