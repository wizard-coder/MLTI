#!/bin/bash
# train(train consistency 시험, randomness)
# 5-shot
for i in {1..5}
do
    python3 main.py --datasource=NCI --num_classes=2 --interpolation_method=mixup --metatrain_iterations=5000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=10 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --trial=$i
done

# train(basic, reproduce)
python3 main.py --datasource=NCI --num_classes=2 --interpolation_method=mixup --metatrain_iterations=5000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=10 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --reproduce



# test(train consistency 시험)
python3 main.py --datasource=NCI --num_classes=2 --interpolation_method=mixup --metatrain_iterations=5000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=10 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=5000 --train_consistency_test=5

# test(test task num consistency 시험)
python3 main.py --datasource=NCI --num_classes=2 --interpolation_method=mixup --metatrain_iterations=5000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=10 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=5000 --trial=4 --task_num_consistency_test 600 2000 4000 6000 8000

# test(basic, reproduce)
python3 main.py --datasource=NCI --num_classes=2 --interpolation_method=mixup --metatrain_iterations=5000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=10 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=5000 --reproduce