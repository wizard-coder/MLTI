import argparse
import random
import numpy as np
import torch
import os
from data_generator import MiniImagenet, ISIC, DermNet, NCI, Metabolism, RainbowMNIST
from learner import Learner
from maml import MAML
import re

parser = argparse.ArgumentParser(description='MLTI')
parser.add_argument('--datasource', default='miniimagenet',
                    type=str, choices=['miniimagenet', 'isic', 'dermnet', 'NCI', 'metabolism', 'rainbowmnist'])
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--test_epoch', default=-1, type=int,
                    help='test epoch, only work when test start')
parser.add_argument('--num_test_task', default=600,
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

print(exp_string)


def train(args, net: MAML, optimiser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc = 0.0, 0.0

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

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > args.metatrain_iterations:
            break

        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
            x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)
        task_losses = []
        task_acc = []

        for meta_batch in range(args.meta_batch_size):

            if args.interpolation_method == "none":
                loss_val, acc_val = net.forward(x_spt[meta_batch], y_spt[meta_batch],
                                                x_qry[meta_batch],
                                                y_qry[meta_batch])
            else:
                mix_c = random.randint(0, 1)

                if mix_c == 1:
                    second_id = (meta_batch + 1) % args.meta_batch_size

                    if args.interpolation_method == "cutmix":
                        loss_val, acc_val = net.forward_cutmix(x_spt[meta_batch], y_spt[meta_batch],
                                                               x_qry[meta_batch],
                                                               y_qry[meta_batch],
                                                               x_spt[second_id], y_spt[second_id],
                                                               x_qry[second_id],
                                                               y_qry[second_id])
                    elif args.interpolation_method == "mixup":
                        loss_val, acc_val = net.forward_mixup(x_spt[meta_batch], y_spt[meta_batch],
                                                              x_qry[meta_batch],
                                                              y_qry[meta_batch],
                                                              x_spt[second_id], y_spt[second_id],
                                                              x_qry[second_id],
                                                              y_qry[second_id])
                else:
                    if args.interpolation_method == "cutmix":
                        loss_val, acc_val = net.forward_cutmix(x_spt[meta_batch], y_spt[meta_batch],
                                                               x_qry[meta_batch],
                                                               y_qry[meta_batch],
                                                               x_spt[meta_batch], y_spt[meta_batch],
                                                               x_qry[meta_batch],
                                                               y_qry[meta_batch])
                    elif args.interpolation_method == "mixup":
                        loss_val, acc_val = net.forward_mixup(x_spt[meta_batch], y_spt[meta_batch],
                                                              x_qry[meta_batch],
                                                              y_qry[meta_batch],
                                                              x_spt[meta_batch], y_spt[meta_batch],
                                                              x_qry[meta_batch],
                                                              y_qry[meta_batch])

            task_losses.append(loss_val)
            task_acc.append(acc_val)

        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        if step != 0 and step % Print_Iter == 0:
            print('{}, {}, {}'.format(step, print_loss, print_acc))
            print_loss, print_acc = 0.0, 0.0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            torch.save(net.net.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, net: MAML):

    net.eval()
    res_acc = []
    args.meta_batch_size = 1

    if args.datasource == 'miniimagenet':
        dataloader = MiniImagenet(args, 'test')
    elif args.datasource == 'isic':
        dataloader = ISIC(args, 'test')
    elif args.datasource == 'dermnet':
        dataloader = DermNet(args, 'test')
    elif args.datasource == 'NCI':
        dataloader = NCI(args, 'test')
    elif args.datasource == 'metabolism':
        dataloader = Metabolism(args, 'test')
    elif args.datasource == 'rainbowmnist':
        dataloader = RainbowMNIST(args, 'test')

    if args.task_num_consistency_test is None:
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
            if step > args.num_test_task:
                break

            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                x_qry.squeeze(0).to(args.device), y_qry.squeeze(
                    0).to(args.device)

            acc = net.finetunning(x_spt, y_spt, x_qry, y_qry)

            res_acc.append(acc.item())

        res_acc = np.array(res_acc)

        print('acc is {}, ci95 is {}'.format(np.mean(res_acc), 1.96 * np.std(res_acc) / np.sqrt(
            args.num_test_task * args.meta_batch_size)))

        return np.mean(res_acc)
    else:
        acc_mean = []
        acc_std = []
        acc_max = []
        acc_min = []
        for task_num in args.task_num_consistency_test:

            acc_mean_per_task_num = []

            for i in range(5):
                res_acc = []
                for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
                    if step > task_num:
                        break

                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                        x_qry.squeeze(0).to(args.device), y_qry.squeeze(
                            0).to(args.device)

                    acc = net.finetunning(x_spt, y_spt, x_qry, y_qry)

                    res_acc.append(acc.item())

                res_acc = np.array(res_acc)

                print('task num is {}, iter is {}, acc is {}, ci95 is {}'.format(task_num, i, np.mean(res_acc), 1.96 * np.std(res_acc) / np.sqrt(
                    task_num * args.meta_batch_size)))

                acc_mean_per_task_num.append(np.mean(res_acc))

            acc_mean_per_task_num = np.array(acc_mean_per_task_num)

            acc_mean.append(np.mean(acc_mean_per_task_num))
            acc_std.append(np.std(acc_mean_per_task_num))
            acc_max.append(np.max(acc_mean_per_task_num))
            acc_min.append(np.min(acc_mean_per_task_num))

        return acc_mean, acc_std, acc_max, acc_min


def main():

    learner: Learner = None
    meta_optim: torch.optim.Adam = None
    maml: MAML = None

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
                    beta_dist_alpha=2, beta_dist_beta=2)
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
                    beta_dist_alpha=2, beta_dist_beta=2)
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
                    beta_dist_alpha=2, beta_dist_beta=2)
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
                    beta_dist_alpha=2, beta_dist_beta=2)

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(
            args.logdir, args.test_epoch, exp_string)
        print(model_file)
        learner.load_state_dict(torch.load(model_file))

    if args.train == 1:
        train(args, maml, meta_optim)
    else:
        if args.train_consistency_test == 0:

            if args.task_num_consistency_test is None:
                model_file = '{0}/{2}/model{1}'.format(
                    args.logdir, args.test_epoch, exp_string)
                maml.net.load_state_dict(torch.load(model_file))
                test(args, maml)
            else:
                model_file = '{0}/{2}/model{1}'.format(
                    args.logdir, args.test_epoch, exp_string)
                maml.net.load_state_dict(torch.load(model_file))
                acc_mean, acc_std, acc_max, acc_min = test(args, maml)

                for task_num, mean, std, max, min in zip(args.task_num_consistency_test, acc_mean, acc_std, acc_max, acc_min):
                    print(
                        f'Task num: {task_num}, mean: {mean}, std: {std}, max: {max}, min: {min}')
        else:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            accs = []
            for i in range(1, args.train_consistency_test + 1):

                new_exp_string = re.sub('trial\d+', f'trial{i}', exp_string)

                model_file = '{0}/{2}/model{1}'.format(
                    args.logdir, args.test_epoch, new_exp_string)
                maml.net.load_state_dict(torch.load(model_file))
                accs.append(test(args, maml))

            print(
                f'mean: {np.mean(accs)}, std: {np.std(accs)}, max: {np.max(accs)}, min: {np.min(accs)}')


if __name__ == '__main__':
    main()
