#!/bin/bash

######################### MLTI
# train(train consistency 시험, randomness)
for i in {1..5}
do
    python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --trial=$i
done

# train(basic, reproduce)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --reproduce

# train(augmentation, reproduce)
# augmix
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --reproduce --augmentation=augmix
# trivial aug
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --reproduce --augmentation=trivialaug


# train(rand conv)
python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5

# test(train consistency 시험)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=15000 --train_consistency_test=5

# test(test task num consistency 시험)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=15000 --trial=2 --task_num_consistency_test 600 2000 4000 6000 8000

# test(basic, reproduce)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=15000 --reproduce

# test(augmentation, reproduce)
# augmix
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=15000 --reproduce --augmentation=augmix
# trivial aug
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=15000 --reproduce --augmentation=trivialaug


# test(rand conv)
python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python main.py --datasource=isic --metatrain_iterations=30000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=1 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.5




######################## vanilla
# train(basic, reproduce)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --device=cuda:1 --reproduce
# train(rand conv, reproduce)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

# test(basic, reproduce)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --train=0 --test_epoch=15000 --reproduce
# test(rand conv, reproduce)
python main.py --datasource=isic --metatrain_iterations=15000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --num_filters=32 --mix=0 --train=0 --test_epoch=15000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

