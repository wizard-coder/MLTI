import argparse
import random
import numpy as np
import torch
import os
from protonet import Protonet
from learner import Conv_Standard
from data_generator import RainbowMNIST
import re
from torchvision import transforms

parser = argparse.ArgumentParser(description='MLTI')
parser.add_argument('--datasource', default='rainbowmnist', type=str,
                    help='rainbowmnist')
parser.add_argument('--num_classes', default=10, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=8000,
                    type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int,
                    help='test epoch, only work when test start')

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
parser.add_argument('--rmnist_extreme_dist', type=str,
                    choices=['color', 'scale', 'rotation', 'compare'])


args = parser.parse_args()
args.datadir = os.path.expanduser(args.datadir)
args.logdir = os.path.expanduser(args.logdir)
print(args)

torch.backends.cudnn.benchmark = True

exp_string = 'ProtoNet_Cross' + '.data_' + str(args.datasource) + 'cls_' + str(args.num_classes) + '.mbs_' + str(
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

if args.rmnist_extreme_dist is not None:
    exp_string += f'.rmnist_extreme_dist_{args.rmnist_extreme_dist}'

if args.reproduce:
    exp_string += '.reproduce'

print(exp_string)

if args.reproduce:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, protonet, optimiser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    transform = None
    if args.augmentation == "augmix":
        transform = transforms.Compose([transforms.AugMix()])
    elif args.augmentation == "trivialaug":
        transform = transforms.Compose(
            [transforms.TrivialAugmentWide()])
    elif args.augmentation == "randaug":
        transform = transforms.Compose(
            [transforms.RandAugment()])

    dataloader = RainbowMNIST(
        args, 'train', transform=transform, extreme_dist=args.rmnist_extreme_dist)

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
            if args.mix:
                mix_c = random.randint(0, 3)
                # 75% cross task interpolation, 25% intra task interpolation
                # 꼭 위의 확률은 아닌게 batch를 구성할때 task를 랜덤하게 뽑기 때문에 하나의 batch안에 동일한 task가 있을수도 있음
                if mix_c < 3:
                    second_id = (meta_batch + 1) % args.meta_batch_size
                    loss_val, acc_val = protonet.forward_crossmix(x_spt[meta_batch], y_spt[meta_batch],
                                                                  x_qry[meta_batch],
                                                                  y_qry[meta_batch],
                                                                  x_spt[second_id], y_spt[second_id],
                                                                  x_qry[second_id],
                                                                  y_qry[second_id])
                else:
                    loss_val, acc_val = protonet.forward_crossmix(x_spt[meta_batch], y_spt[meta_batch],
                                                                  x_qry[meta_batch],
                                                                  y_qry[meta_batch],
                                                                  x_spt[meta_batch], y_spt[meta_batch],
                                                                  x_qry[meta_batch],
                                                                  y_qry[meta_batch])
            else:
                loss_val, acc_val = protonet(
                    x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])
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
            torch.save(protonet.learner.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, protonet):
    res_acc = []
    args.meta_batch_size = 1

    dataloader = RainbowMNIST(args, 'test')

    if args.task_num_consistency_test is None:
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
            if step > args.num_test_task:
                break
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                x_qry.squeeze(0).to(args.device), y_qry.squeeze(
                    0).to(args.device)
            _, acc_val = protonet(x_spt, y_spt, x_qry, y_qry)
            res_acc.append(acc_val.item())

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
                    _, acc_val = protonet(x_spt, y_spt, x_qry, y_qry)
                    res_acc.append(acc_val.item())

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
    # base learner
    learner = Conv_Standard(
        args=args, x_dim=3, hid_dim=args.num_filters, z_dim=args.num_filters).to(args.device)

    rand_conv = args.augmentation == 'randconv'

    # protonet
    protonet = Protonet(args, learner, rand_conv=rand_conv,
                        rand_conv_prob=args.randconv_prob, rand_conv_mixing=args.randconv_mix)

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(
            args.logdir, args.test_epoch, exp_string)
        print(model_file)
        learner.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(learner.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, protonet, meta_optimiser)
    else:
        if args.train_consistency_test == 0:

            if args.task_num_consistency_test is None:
                model_file = '{0}/{2}/model{1}'.format(
                    args.logdir, args.test_epoch, exp_string)
                protonet.learner.load_state_dict(torch.load(model_file))
                test(args, protonet)
            else:
                model_file = '{0}/{2}/model{1}'.format(
                    args.logdir, args.test_epoch, exp_string)
                protonet.learner.load_state_dict(torch.load(model_file))
                acc_mean, acc_std, acc_max, acc_min = test(args, protonet)

                for task_num, mean, std, max, min in zip(args.task_num_consistency_test, acc_mean, acc_std, acc_max, acc_min):
                    print(
                        f'Task num: {task_num}, mean: {mean}, std: {std}, max: {max}, min: {min}, max-min: {max-min}')
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
                protonet.learner.load_state_dict(torch.load(model_file))
                accs.append(test(args, protonet))

            print(
                f'mean: {np.mean(accs)}, std: {np.std(accs)}, max: {np.max(accs)}, min: {np.min(accs)}, max-min: {np.max(accs)-np.min(accs)}')


if __name__ == '__main__':
    main()
