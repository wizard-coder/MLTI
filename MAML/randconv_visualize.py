from data_generator import MiniImagenet
import argparse
from rand_conv import RandConvModule
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as F
import numpy as np
import torch

parser = argparse.ArgumentParser(description='MLTI')
parser.add_argument('--datasource', default='miniimagenet',
                    type=str, choices=['miniimagenet', 'isic', 'dermnet', 'NCI', 'metabolism', 'rainbowmnist'])
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--test_epoch', default=-1, type=int,
                    help='test epoch, only work when test start')
parser.add_argument('--num_test_task', default=8000,
                    type=int, help='number of test tasks.')
parser.add_argument('--interpolation_method', default="cutmix",
                    type=str, choices=['cutmix', 'mixup', 'none'])

# Training options
parser.add_argument('--metatrain_iterations', default=15000, type=int,
                    help='number of metatraining iterations.')
parser.add_argument('--meta_batch_size', default=25, type=int,
                    help='number of tasks sampled per meta-update')
parser.add_argument('--meta_lr', default=0.001,
                    type=float, help='outer learning rate')
parser.add_argument('--update_lr', default=0.01,
                    type=float, help='inner learning rate')
parser.add_argument('--update_step', default=5,
                    type=int, help='inner update steps')
parser.add_argument('--update_step_test', default=10,
                    type=int, help='inner update steps for test')
parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')

# Logging, saving etc
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str,
                    help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int,
                    help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int,
                    help='True to train, False to test.')
parser.add_argument('--trial', default=0, type=int,
                    help='trail for each layer')
parser.add_argument('--ratio', default=1.0, type=float,
                    help='the ratio of meta-training tasks')
parser.add_argument("--device", default='cuda:0', type=str,
                    help="cuda:num or mps:num or cpu")

# experiment
parser.add_argument('--train_consistency_test', type=int, default=0, choices=range(0, 100),
                    help='학습마다 성능차이가 얼마나는지 테스트')
parser.add_argument('--task_num_consistency_test', type=int, nargs='+',
                    help='test시 tesk num 마다 성능차이 시험')
parser.add_argument('--augmentation', type=str,
                    choices=['augmix', 'trivialaug', 'randaug', 'randconv'])
parser.add_argument('--reproduce', action='store_true',
                    help='reproduce the result')
parser.add_argument('--randconv_prob', type=float,
                    default=0, help='randconv prob')
parser.add_argument('--randconv_mix', action='store_true', help='randconv mix')
parser.add_argument('--randconv_multi_std', type=float, nargs='+')
parser.add_argument('--randconv_only_support', action='store_true')
parser.add_argument('--rmnist_extreme_dist', type=str,
                    choices=['color', 'scale', 'rotation', 'compare'])

args = parser.parse_args()
args.datadir = os.path.expanduser(args.datadir)
args.logdir = os.path.expanduser(args.logdir)

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def main():
    dataloader = MiniImagenet(args, 'train')

    x, _, _, _ = dataloader[0]
    x = x.to(args.device)


    x1 = x[0][0].unsqueeze(0)
    x2 = x[1][0].unsqueeze(0)

    num_image = 5
    img_list1 = [x1.squeeze(0)]
    img_list2 = [x2.squeeze(0)]

    for i in range(num_image):
        rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                        in_channels=3,
                                        out_channels=3,
                                        mixing=False,
                                        identity_prob=0.0,
                                        ).to(args.device)
        
        randconv_x1 = rand_conv(x1)
        randconv_x2 = rand_conv(x2)

        randconv_x1 = (randconv_x1 - torch.min(randconv_x1)) / (torch.max(randconv_x1) - torch.min(randconv_x1))
        randconv_x2 = (randconv_x2 - torch.min(randconv_x2)) / (torch.max(randconv_x2) - torch.min(randconv_x2))


        img_list1.append(randconv_x1.squeeze(0))
        img_list2.append(randconv_x2.squeeze(0))

    img_list = img_list1 + img_list2
    grid = make_grid(img_list, num_image+1, normalize=False)
    show(grid)
    print("done")
    plt.show()

if __name__ == "__main__":
    main()
