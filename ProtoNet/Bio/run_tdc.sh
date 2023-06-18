#!/bin/bash

# train(train consistency 시험, randomness)
for i in {1..5}
do
    python main.py --datasource=metabolism --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=10 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --mix=1 --device=cuda:1 --trial=$i
done

# train(basic, reproduce)
python main.py --datasource=metabolism --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=10 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --mix=1 --device=cuda:1 --reproduce



# test(train consistency 시험)
python main.py --datasource=metabolism --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=10 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --mix=1 --train=0 --test_epoch=20000 --train_consistency_test=5

# test(test task num consistency 시험)
python main.py --datasource=metabolism --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=10 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --mix=1 --train=0 --test_epoch=20000 --trial=1 --task_num_consistency_test 600 2000 4000 6000 8000

# test(basic, reproduce)
python main.py --datasource=metabolism --metatrain_iterations=20000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=10 --num_classes=2 --datadir=~/research/data/MLTI/ --logdir=./logs --mix=1 --train=0 --test_epoch=20000 --reproduce