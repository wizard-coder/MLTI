import argparse
import random
import numpy as np
import torch
import os
from data_generator import MiniImagenet, ISIC, DermNet, NCI, Metabolism, RainbowMNIST
from learner import Learner
from maml import MAML
import re
from torchvision import transforms
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

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
print(args)

torch.backends.cudnn.benchmark = True

exp_string = 'MAML_Cross' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.intlmethod_' + str(args.interpolation_method) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr) + '.updatelr' + str(args.update_lr) + '.updatestep' + str(args.update_step)


exp_string += '.trial{}'.format(args.trial)
if args.ratio < 1.0:
    exp_string += '.ratio{}'.format(args.ratio)

if args.augmentation is not None:
    exp_string += f'.aug_{args.augmentation}'

if args.augmentation == 'randconv':
    exp_string += f'.randconv_prob_{args.randconv_prob}'
    if args.randconv_mix:
        exp_string += '.randconv_mix'
    if args.randconv_multi_std is not None:
        std = ''

        for s in args.randconv_multi_std:
            std += f'_{s}'

        exp_string += '.randconv_multi_std' + std

    if args.randconv_only_support:
        exp_string += '.randconv_only_support'

if args.rmnist_extreme_dist is not None:
    exp_string += f'.rmnist_extreme_dist_{args.rmnist_extreme_dist}'

if args.reproduce:
    exp_string += '.reproduce'

print(exp_string)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    learner: Learner = None
    meta_optim: torch.optim.Adam = None
    maml: MAML = None

    rand_conv = args.augmentation == 'randconv'

    if args.datasource == 'miniimagenet':
        config = [
            [('conv2d', [32, 3, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('flatten', [])],
            [('linear', [args.num_classes, 32 * 5 * 5])]
        ]
        learner = Learner(config=config).to(args.device)

        meta_optim = torch.optim.Adam(learner.parameters(), lr=args.meta_lr)

        maml = MAML(args, learner, label_sharing=False, interpolation_block_max=0,
                    beta_dist_alpha=2, beta_dist_beta=2, rand_conv=rand_conv, rand_conv_prob=args.randconv_prob, rand_conv_mixing=args.randconv_mix,
                    rand_conv_multi_std=args.randconv_multi_std, rand_conv_only_support=args.randconv_only_support)
    elif args.datasource == 'isic':
        config = [
            [('conv2d', [32, 3, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('flatten', [])],
            [('linear', [args.num_classes, 32 * 5 * 5])]
        ]
        learner = Learner(config=config).to(args.device)

        meta_optim = torch.optim.Adam(learner.parameters(), lr=args.meta_lr)

        maml = MAML(args, learner, label_sharing=False, interpolation_block_max=3,
                    beta_dist_alpha=2, beta_dist_beta=2, rand_conv=rand_conv, rand_conv_prob=args.randconv_prob, rand_conv_mixing=args.randconv_mix,
                    rand_conv_multi_std=args.randconv_multi_std, rand_conv_only_support=args.randconv_only_support)
    elif args.datasource == 'dermnet':
        config = [
            [('conv2d', [32, 3, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('flatten', [])],
            [('linear', [args.num_classes, 32 * 5 * 5])]
        ]
        learner = Learner(config=config).to(args.device)

        meta_optim = torch.optim.Adam(learner.parameters(), lr=args.meta_lr)

        maml = MAML(args, learner, label_sharing=False, interpolation_block_max=3,
                    beta_dist_alpha=2, beta_dist_beta=2, rand_conv=rand_conv, rand_conv_prob=args.randconv_prob, rand_conv_mixing=args.randconv_mix,
                    rand_conv_multi_std=args.randconv_multi_std, rand_conv_only_support=args.randconv_only_support)
    elif args.datasource == 'NCI':
        config = [
            [('linear', [500, 1024]),
             ('bn', [500]),
             ('leakyrelu', [0.01, True])],
            [('linear', [500, 500]),
             ('bn', [500]),
             ('leakyrelu', [0.01, True])],
            [('linear', [args.num_classes, 500])]
        ]

        learner = Learner(config=config).to(args.device)

        meta_optim = torch.optim.Adam(learner.parameters(), lr=args.meta_lr)

        maml = MAML(args, learner, label_sharing=True, interpolation_block_max=1,
                    beta_dist_alpha=2, beta_dist_beta=2)

    elif args.datasource == 'metabolism':
        config = [
            [('linear', [500, 1024]),
             ('bn', [500]),
             ('leakyrelu', [0.01, True])],
            [('linear', [500, 500]),
             ('bn', [500]),
             ('leakyrelu', [0.01, True])],
            [('linear', [args.num_classes, 500])]
        ]

        learner = Learner(config=config).to(args.device)

        meta_optim = torch.optim.Adam(learner.parameters(), lr=args.meta_lr)

        maml = MAML(args, learner, label_sharing=True, interpolation_block_max=1,
                    beta_dist_alpha=0.5, beta_dist_beta=0.5)

    elif args.datasource == 'rainbowmnist':
        config = [
            [('conv2d', [32, 3, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('conv2d', [32, 32, 3, 3, 1, 1]),
             ('bn', [32]),
             ('relu', [True]),
             ('max_pool2d', [2, 2, 0])],
            [('flatten', [])],
            [('linear', [args.num_classes, 32 * 1 * 1])]
        ]
        learner = Learner(config=config).to(args.device)

        meta_optim = torch.optim.Adam(learner.parameters(), lr=args.meta_lr)

        maml = MAML(args, learner, label_sharing=True, interpolation_block_max=3,
                    beta_dist_alpha=2, beta_dist_beta=2, rand_conv=rand_conv, rand_conv_prob=args.randconv_prob, rand_conv_mixing=args.randconv_mix,
                    rand_conv_multi_std=args.randconv_multi_std, rand_conv_only_support=args.randconv_only_support)

    if args.datasource == 'miniimagenet':
        dataloader = MiniImagenet(args, 'train')
    elif args.datasource == 'isic':
        dataloader = ISIC(args, 'train')
    elif args.datasource == 'dermnet':
        dataloader = DermNet(args, 'train')
    elif args.datasource == 'NCI':
        dataloader = NCI(args, 'train')
    elif args.datasource == 'metabolism':
        dataloader = Metabolism(args, 'train')
    elif args.datasource == 'rainbowmnist':
        dataloader = RainbowMNIST(args, 'train')

    x, y, _, _ = dataloader[0]
    x = x.to(args.device)
    y = y.to(args.device)

    grad, label = maml.get_grad(
        x[0], y[0], x[1], y[1], num_samples=1000, grad_layer=8)

    print('complete grad')

    tsne = TSNE(n_components=2, random_state=0)
    cluster = np.array(tsne.fit_transform(np.array(grad)))
    label = np.array(label)

    print('complete tsne')

    label_list = ['x1_grad', 'x2_grad', 'x1_randconv_grad',
                  'x2_randconv_grad', 'mlti_grad', 'mlti_with_randconv_grad']

    for l in label_list:
        idx = np.where(label == l)

        if l == 'x1_grad':
            plt.scatter(cluster[idx, 0], cluster[idx, 1],
                        s=300, c='r', marker=(5, 1), label='Task 1')
        elif l == 'x2_grad':
            plt.scatter(cluster[idx, 0], cluster[idx, 1],
                        s=300, c='b', marker=(5, 1), label='Task 2')
        elif l == 'x1_randconv_grad':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='r', marker='+', label='Task 1 with TARC')
        elif l == 'x2_randconv_grad':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='b', marker='+', label='Task 2 with TARC')
        elif l == 'mlti_grad':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='g', label='MLTI')
        elif l == 'mlti_with_randconv_grad':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='m', marker='>', label='MLTI with TARC')

    plt.legend(loc='upper left', frameon=False)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
