# train
# 1-shot
# for i in {1..5}
# do
#     python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --trial=$i
# done
# 5-shot
for i in {1..5}
do
    python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --device=cuda:1 --trial=$i
done
# test
# 1-shot
# python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=30000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=1 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=30000
# 5-shot
# python3 main.py --datasource=isic --num_classes=2 --interpolation_method=cutmix --metatrain_iterations=15000 --meta_batch_size=4 --meta_lr=0.001 --update_lr=0.01 --update_step=5 --update_step_test=5 --update_batch_size=5 --update_batch_size_eval=15 --logdir=./logs --datadir=~/research/data/MLTI/ --train=0 --test_epoch=15000
