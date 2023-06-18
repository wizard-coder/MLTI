#!/bin/bash

######################### MLTI
# train(train consistency 시험, randomness)
for i in {1..5}
do
    python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --trial=$i
done

# train(basic, reproduce)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce

# train(augment, reproduce)
# augmix
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --augmentation=augmix
# trivial aug
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --augmentation=trivialaug


# train(rand conv)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5


# train(rmnist extream dist)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=color

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=color

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=scale

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=scale

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=rotation

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=rotation

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=compare

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --device=cuda:1 --reproduce --rmnist_extreme_dist=compare



# test(train consistency 시험)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --train=0 --test_epoch=30000 --train_consistency_test=5

# test(test task num consistency 시험) 
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --train=0 --test_epoch=30000 --trial=3 --task_num_consistency_test 600 2000 4000 6000 8000

# test(basic, reproduce)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce

# test(augment, reproduce)
# augmix
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --augmentation=augmix
# trivial aug
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --augmentation=trivialaug



# test(rand conv)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.5


# test(rmnist extream dist)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=color

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=color

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=scale

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=scale

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=rotation

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=rotation

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=compare

python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --test_epoch=30000 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=compare



######################## vanilla
# train(basic, reproduce)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --device=cuda:1 --reproduce
# train(rand conv, reproduce)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

# test(basic, reproduce)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --train=0 --test_epoch=30000 --reproduce
# test(rand conv, reproduce)
python main.py --datasource=rainbowmnist --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=10 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix
