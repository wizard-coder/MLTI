#!/bin/bash

######################### MLTI
# train(train consistency 시험, randomness)
for i in {1..5}
do
    python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --trial=$i
done

# train(basic, reproduce)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce

# train(augmentation, reproduce)
# augmix
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --augmentation=augmix
# trivial aug
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --augmentation=trivialaug


# train(rand conv)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.5

# train(rmnist extream dist)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=color

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=color

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=scale

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=scale

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=rotation

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=rotation

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=compare

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --rmnist_extreme_dist=compare

# test(train consistency 시험)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --train_consistency_test=5

# test(test task num consistency 시험)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --trial=2 --task_num_consistency_test 600 2000 4000 6000 8000

# test(basic, reproduce)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce

# test(augmentation, reproduce)
# augmix
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --augmentation=augmix
# trivial aug
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --augmentation=trivialaug

# test(rand conv)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.5 --randconv_mix

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.0 --randconv_mix

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.5


# test(rmnist extream dist)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=color

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=color

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=scale

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=scale

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=rotation

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=rotation

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=compare

python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --rmnist_extreme_dist=compare


######################## vanilla
# train(basic, reproduce)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce
# train(rand conv, reproduce)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --device=cuda:0 --reproduce --augmentation=randconv --randconv_prob=0.5

# test(basic, reproduce)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce
# test(rand conv, reproduce)
python3 main.py --datasource=rainbowmnist --num_classes=10 --interpolation_method=none --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --ratio=0.4 --train=0 --test_epoch=30000 --reproduce --augmentation=randconv --randconv_prob=0.5
