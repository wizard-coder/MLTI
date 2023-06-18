#!/bin/bash

######################### MLTI
# train(train consistency 시험, randomness)
for i in {1..5}
do
    python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --trial=$i
done

# train(basic, reproduce)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce

# train(augmentation, reproduce)
# augmix
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce --augmentation=augmix
# trivial aug
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce --augmentation=trivialaug

# train(rand conv)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5


# test(train consistency 시험)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --train_consistency_test=5

# test(test task num consistency 시험)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --trial=1 --task_num_consistency_test 600 2000 4000 6000 8000

# test(basic, reproduce)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce

# test(augmentation, reproduce)
# augmix
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce --augmentation=augmix
# trivial aug
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce --augmentation=trivialaug


# test(rand conv)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=10 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce --augmentation=randconv --randconv_prob=0.5






######################## vanilla
# train(basic, reproduce)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=none --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce
# train(randconv, reproduce)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=none --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

# test(basic, reproduce)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=none --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce
# test(randconv, reproduce)
python3 main.py --datasource=isic --num_classes=2 --interpolation_method=none --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=5 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix
