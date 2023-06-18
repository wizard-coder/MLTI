import argparse
import random
import numpy as np
import torch
import os
from data_generator import MiniImagenet, ISIC, DermNet
from learner import Conv_Standard
from protonet import Protonet
import re
from torchvision import transforms
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='MLTI')
parser.add_argument('--datasource', default='miniimagenet', type=str,
                    help='miniimagenet, isic, dermnet')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--test_epoch', default=-1, type=int,
                    help='test epoch, only work when test start')
parser.add_argument('--num_test_task', default=8000,
                    type=int, help='number of test tasks.')

# Training options
parser.add_argument('--metatrain_iterations', default=15000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=25, type=int,
                    help='number of tasks sampled per meta-update')
parser.add_argument('--meta_lr', default=0.001, type=float,
                    help='the base learning rate of the generator')
parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')

# Model options
parser.add_argument('--num_filters', default=64, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0,
                    type=float, help='weight decay')

# Logging, saving, and testing options
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str,
                    help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int,
                    help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int,
                    help='True to train, False to test.')
parser.add_argument('--mix', default=0, type=int, help='use mixup or not')
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


args = parser.parse_args()
args.datadir = os.path.expanduser(args.datadir)
args.logdir = os.path.expanduser(args.logdir)
print(args)

if args.datasource == 'isic':
    assert args.num_classes < 5

torch.backends.cudnn.benchmark = True


exp_string = 'ProtoNet_Cross' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr)

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.mix:
    exp_string += '.mix'
exp_string += '.trial{}'.format(args.trial)
if args.ratio < 1.0:
    exp_string += '.ratio{}'.format(args.ratio)
if args.augmentation is not None:
    exp_string += f'.aug_{args.augmentation}'

if args.augmentation == 'randconv':
    exp_string += f'.randconv_prob_{args.randconv_prob}'
    if args.randconv_mix:
        exp_string += '.randconv_mix'

if args.reproduce:
    exp_string += '.reproduce'

print(exp_string)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    learner = Conv_Standard(
        args=args, x_dim=3, hid_dim=args.num_filters, z_dim=args.num_filters).to(args.device)

    rand_conv = args.augmentation == 'randconv'

    protonet = Protonet(args, learner, rand_conv=rand_conv,
                        rand_conv_prob=args.randconv_prob, rand_conv_mixing=args.randconv_mix)

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(
            args.logdir, args.test_epoch, exp_string)
        print(model_file)
        learner.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(learner.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.datasource == 'miniimagenet':
        dataloader = MiniImagenet(args, 'train')
    elif args.datasource == 'isic':
        dataloader = ISIC(args, 'train')
    elif args.datasource == 'dermnet':
        dataloader = DermNet(args, 'train')

    x, y, _, _ = dataloader[0]
    x = x.to(args.device)
    y = y.to(args.device)

    proto, label = protonet.get_proto(x[0], x[1], num_samples=1000)

    print('compute proto')

    tsne = TSNE(n_components=2, random_state=0)
    cluster = np.array(tsne.fit_transform(np.array(proto)))
    label = np.array(label)

    print('compute tsne')

    label_list = ['x1_proto', 'x2_proto', 'x1_randconv_proto',
                  'x2_randconv_proto', 'mlti_proto', 'mlti_randconv_proto']

    for l in label_list:
        idx = np.where(label == l)

        if l == 'x1_proto':
            plt.scatter(cluster[idx, 0], cluster[idx, 1],
                        s=300, c='r', marker=(5, 1), label='Task 1')
        elif l == 'x2_proto':
            plt.scatter(cluster[idx, 0], cluster[idx, 1],
                        s=300, c='b', marker=(5, 1), label='Task 2')
        elif l == 'x1_randconv_proto':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='r', marker='+', label='Task 1 with TARC')
        elif l == 'x2_randconv_proto':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='b', marker='+', label='Task 2 with TARC')
        elif l == 'mlti_proto':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='g', label='MLTI')
        elif l == 'mlti_randconv_proto':
            plt.scatter(cluster[idx, 0], cluster[idx, 1], c='m', marker='>', label='MLTI with TARC')

    plt.legend(loc='upper left', frameon=False)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
