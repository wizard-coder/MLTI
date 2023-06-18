import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from typing import Tuple
from torch.distributions import Beta

from copy import deepcopy
from enum import Enum
from learner import Learner
from rand_conv import RandConvModule


class MAML(nn.Module):
    """
    MAML Learner
    """

    def __init__(self, args, learner: Learner, label_sharing=False, interpolation_block_max=0, beta_dist_alpha=2.0, beta_dist_beta=2.0,
                 rand_conv=False, rand_conv_prob=0.0, rand_conv_mixing=False, rand_conv_multi_std=None, rand_conv_only_support=False):
        """

        :param args:
        """
        super(MAML, self).__init__()

        self.args = args
        self.update_lr = args.update_lr  # inner loop learning rate
        self.meta_lr = args.meta_lr  # outer loop learning rate
        self.update_step = args.update_step  # inner loop update steps
        # inner loop update steps for testing
        self.update_step_test = args.update_step_test
        self.device = args.device

        self.net = learner
        self.label_sharing = label_sharing
        self.interpolation_block_max = interpolation_block_max
        self.rand_conv = rand_conv
        self.rand_conv_prob = rand_conv_prob
        self.rand_conv_mixing = rand_conv_mixing
        self.rand_conv_multi_std = rand_conv_multi_std
        self.rand_conv_only_support = rand_conv_only_support
        print(self.rand_conv)

        self.dist = Beta(torch.FloatTensor(
            [beta_dist_alpha]), torch.FloatTensor([beta_dist_beta]))

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam.cpu())
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix(self, xs, xq, lam):
        mixed_x = xq.clone()
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(xq.size(), lam)

        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = xs[:, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (xq.size()[-1] * xq.size()[-2]))

        return mixed_x, lam

    def manifold_mixup(self, xs, xq, lam):
        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, lam

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        if self.rand_conv:
            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob,
                                       multi_std=self.rand_conv_multi_std).to(self.device)

            x_spt = rand_conv(x_spt).detach()

            if not self.rand_conv_only_support:
                x_qry = rand_conv(x_qry).detach()

        # 1. run the i-th task and compute loss for k=0
        logits = self.net(x_spt, vars=None, bn_training=True)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(
            loss, self.net.parameters(), create_graph=True)
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        for k in range(1, self.update_step):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = self.net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        # end of all tasks
        # sum over all losses on query set across all tasks
        logit_q = self.net(x_qry, fast_weights, bn_training=True)
        loss_q = F.cross_entropy(logit_q, y_qry)

        acc = None
        with torch.no_grad():
            pred_q = F.softmax(logit_q, dim=1).argmax(dim=1)
            acc = torch.eq(pred_q, y_qry).float().mean()

        return loss_q, acc

    def forward_cutmix(self, x1s: torch.Tensor, y1s: torch.Tensor, x1q: torch.Tensor, y1q: torch.Tensor,
                       x2s: torch.Tensor, y2s: torch.Tensor, x2q: torch.Tensor, y2q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x1s (_type_): (setsz, c_, h, w)
            y1s (_type_): _description_
            x1q (_type_): _description_
            y1q (_type_): _description_
            x2s (_type_): _description_
            y2s (_type_): _description_
            x2q (_type_): _description_
            y2q (_type_): _description_
        """

        # 1. task interpolation
        # lamda for interpolation
        lam_mix = self.dist.sample().to(self.device)

        # task1과 task2를 random하게 섞기위해 task2 suffling
        task_2_shuffle_id = np.arange(self.args.num_classes)
        np.random.shuffle(task_2_shuffle_id)
        task_2_shuffle_id_s = np.array(
            [np.arange(self.args.update_batch_size) + task_2_shuffle_id[idx] * self.args.update_batch_size for idx in
             range(self.args.num_classes)]).flatten()
        task_2_shuffle_id_q = np.array(
            [np.arange(self.args.update_batch_size_eval) + task_2_shuffle_id[idx] * self.args.update_batch_size_eval for
             idx in range(self.args.num_classes)]).flatten()

        x2s = x2s[task_2_shuffle_id_s]
        x2q = x2q[task_2_shuffle_id_q]

        x_mix_s, lam_s = self.cutmix(x1s, x2s, lam_mix)
        x_mix_q, lam_q = self.cutmix(x1q, x2q, lam_mix)

        # non label sharing case의 경우 label을 새롭게 할당함. 간단하게 task1의 label을 그대로 사용
        if self.label_sharing:
            y1s = F.one_hot(y1s, num_classes=self.args.num_classes)
            y2s = F.one_hot(y2s, num_classes=self.args.num_classes)
            y1q = F.one_hot(y1q, num_classes=self.args.num_classes)
            y2q = F.one_hot(y2q, num_classes=self.args.num_classes)
            y_mix_s, _ = self.manifold_mixup(y1s, y2s, lam_s)
            y_mix_q, _ = self.manifold_mixup(y1q, y2q, lam_q)
        else:
            y_mix_s = y1s
            y_mix_q = y1q

        if self.rand_conv:
            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob,
                                       multi_std=self.rand_conv_multi_std).to(self.device)

            x_mix_s = rand_conv(x_mix_s).detach()

            if not self.rand_conv_only_support:
                x_mix_q = rand_conv(x_mix_q).detach()

        # inner optimization
        # first iteration
        logits = self.net(x_mix_s, vars=None, bn_training=True)

        if self.label_sharing:
            loss = F.binary_cross_entropy(F.softmax(logits, dim=1), y_mix_s)
        else:
            loss = F.cross_entropy(logits, y_mix_s)

        grad = torch.autograd.grad(
            loss, self.net.parameters(), create_graph=True)
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        for k in range(1, self.update_step):
            # 1. compute loss
            logits = self.net(x_mix_s, fast_weights, bn_training=True)
            if self.label_sharing:
                loss = F.binary_cross_entropy(
                    F.softmax(logits, dim=1), y_mix_s)
            else:
                loss = F.cross_entropy(logits, y_mix_s)

            # 2. compute grad
            grad = torch.autograd.grad(loss, fast_weights, create_graph=True)

            # 3. update weight
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        # outer optimization
        # 1. compute loss
        logits_q = self.net(x_mix_q, fast_weights, bn_training=True)
        if self.label_sharing:
            loss_q = F.binary_cross_entropy(
                F.softmax(logits_q, dim=1), y_mix_q)
        else:
            loss_q = F.cross_entropy(logits_q, y_mix_q)

        # calculate accuracy
        acc = None
        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            if self.label_sharing:
                acc = torch.eq(pred_q, y_mix_q.argmax(dim=1)).float().mean()
            else:
                acc = torch.eq(pred_q, y_mix_q).float().mean()

        return loss_q, acc

    def forward_mixup(self, x1s: torch.Tensor, y1s: torch.Tensor, x1q: torch.Tensor, y1q: torch.Tensor,
                      x2s: torch.Tensor, y2s: torch.Tensor, x2q: torch.Tensor, y2q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # lamda for interpolation
        lam_mix = self.dist.sample().to(self.device)

        # task1과 task2를 random하게 섞기위해 task2 suffling
        task_2_shuffle_id = np.arange(self.args.num_classes)
        np.random.shuffle(task_2_shuffle_id)
        task_2_shuffle_id_s = np.array(
            [np.arange(self.args.update_batch_size) + task_2_shuffle_id[idx] * self.args.update_batch_size for idx in
             range(self.args.num_classes)]).flatten()
        task_2_shuffle_id_q = np.array(
            [np.arange(self.args.update_batch_size_eval) + task_2_shuffle_id[idx] * self.args.update_batch_size_eval for
             idx in range(self.args.num_classes)]).flatten()

        x2s = x2s[task_2_shuffle_id_s]
        x2q = x2q[task_2_shuffle_id_q]

        # non label sharing case의 경우 label을 새롭게 할당함. 간단하게 task1의 label을 그대로 사용
        if self.label_sharing:
            y1s = F.one_hot(y1s, num_classes=self.args.num_classes)
            y2s = F.one_hot(y2s, num_classes=self.args.num_classes)
            y1q = F.one_hot(y1q, num_classes=self.args.num_classes)
            y2q = F.one_hot(y2q, num_classes=self.args.num_classes)
            y_mix_s, _ = self.manifold_mixup(y1s, y2s, lam_mix)
            y_mix_q, _ = self.manifold_mixup(y1q, y2q, lam_mix)
        else:
            y_mix_s = y1s
            y_mix_q = y1q

        # inner optimization
        # first iteration
        # task interpolation
        block_idx = np.random.randint(0, self.interpolation_block_max + 1)
        # freeze layers below interpolation layer
        hidden1_s: torch.Tensor = self.net(
            x1s, vars=None, bn_training=True, partial_block=[0, block_idx])
        hidden2_s: torch.Tensor = self.net(
            x2s, vars=None, bn_training=True, partial_block=[0, block_idx])
        hidden1_q: torch.Tensor = self.net(
            x1q, vars=None, bn_training=True, partial_block=[0, block_idx])
        hidden2_q: torch.Tensor = self.net(
            x2q, vars=None, bn_training=True, partial_block=[0, block_idx])

        x_mix_s, _ = self.manifold_mixup(
            hidden1_s, hidden2_s, lam_mix)
        x_mix_q, _ = self.manifold_mixup(
            hidden1_q, hidden2_q, lam_mix)
        # x_mix_s = x_mix_s.detach()
        # x_mix_q = x_mix_q.detach()

        logits = self.net(x_mix_s, vars=None,
                          bn_training=True, partial_block=[block_idx, -1])

        if self.label_sharing:
            loss = F.binary_cross_entropy(F.softmax(logits, dim=1), y_mix_s)
        else:
            loss = F.cross_entropy(logits, y_mix_s)

        grad = torch.autograd.grad(
            loss, self.net.parameters(), allow_unused=True, create_graph=True)

        fast_weight = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad, self.net.parameters())))

        for k in range(1, self.update_step):
            # TODO: 매번 update 할때마다 task interpolation????
            # 1. compute loss
            logits = self.net(x_mix_s, fast_weight,
                              bn_training=True, partial_block=[block_idx, -1])
            if self.label_sharing:
                loss = F.binary_cross_entropy(
                    F.softmax(logits, dim=1), y_mix_s)
            else:
                loss = F.cross_entropy(logits, y_mix_s)

            # 2. compute grad
            grad = torch.autograd.grad(
                loss, fast_weight, allow_unused=True, create_graph=True)

            # 3. update weight
            fast_weight = list(
                map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad, fast_weight)))

        # outer optimization
        # 1. compute loss
        logits_q = self.net(x_mix_q,  fast_weight,
                            bn_training=True, partial_block=[block_idx, -1])
        if self.label_sharing:
            loss_q = F.binary_cross_entropy(
                F.softmax(logits_q, dim=1), y_mix_q)
        else:
            loss_q = F.cross_entropy(logits_q, y_mix_q)

        # calculate accuracy
        acc = None
        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            if self.label_sharing:
                acc = torch.eq(pred_q, y_mix_q.argmax(dim=1)).float().mean()
            else:
                acc = torch.eq(pred_q, y_mix_q).float().mean()

        return loss_q, acc

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        assert len(x_spt.shape) == 4 or len(x_spt.shape) == 2

        querysz = x_qry.size(0)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        logits_q = net(x_qry, fast_weights, bn_training=True)

        acc = None
        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            acc = torch.eq(pred_q, y_qry).float().mean()

        del net

        return acc

    def get_grad(self, x1, y1, x2, y2, num_samples=100, grad_layer=0):
        # gradient 분포 분석용 코드

        total_grad = []
        total_label = []

        net = deepcopy(self.net)
        logits = net(x1)
        loss = F.cross_entropy(logits, y1)

        grad = torch.autograd.grad(loss, net.parameters())

        total_grad.append(grad[grad_layer].flatten().cpu().numpy())
        total_label.append('x1_grad')

        net = deepcopy(self.net)
        logits = net(x2)
        loss = F.cross_entropy(logits, y2)

        grad = torch.autograd.grad(loss, net.parameters())

        total_grad.append(grad[grad_layer].flatten().cpu().numpy())
        total_label.append('x2_grad')

        # rand conv
        for _ in range(num_samples):

            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob,
                                       multi_std=self.rand_conv_multi_std).to(self.device)

            x1_randconv = rand_conv(x1).detach()
            x2_randconv = rand_conv(x2).detach()

            net = deepcopy(self.net)
            logits = net(x1_randconv)
            loss = F.cross_entropy(logits, y1)

            grad = torch.autograd.grad(loss, net.parameters())

            total_grad.append(
                grad[grad_layer].flatten().cpu().numpy())
            total_label.append('x1_randconv_grad')

            net = deepcopy(self.net)
            logits = net(x2_randconv)
            loss = F.cross_entropy(logits, y2)

            grad = torch.autograd.grad(loss, net.parameters())

            total_grad.append(
                grad[grad_layer].flatten().cpu().numpy())
            total_label.append('x2_randconv_grad')

        # mlti
        for _ in range(num_samples):

            # lamda for interpolation
            lam_mix = self.dist.sample().to(self.device)

            # task1과 task2를 random하게 섞기위해 task2 suffling
            task_2_shuffle_id = np.arange(self.args.num_classes)
            np.random.shuffle(task_2_shuffle_id)
            task_2_shuffle_id_s = np.array(
                [np.arange(self.args.update_batch_size) + task_2_shuffle_id[idx] * self.args.update_batch_size for idx in
                 range(self.args.num_classes)]).flatten()

            x2_shuffle = x2[task_2_shuffle_id_s]

            x_mix_s, lam_s = self.cutmix(x1, x2_shuffle, lam_mix)

            net = deepcopy(self.net)
            logits = net(x_mix_s)
            loss = F.cross_entropy(logits, y1)

            grad = torch.autograd.grad(loss, net.parameters())

            total_grad.append(
                grad[grad_layer].flatten().cpu().numpy())
            total_label.append('mlti_grad')

            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob,
                                       multi_std=self.rand_conv_multi_std).to(self.device)

            x_mix_s_randconv = rand_conv(x_mix_s).detach()

            net = deepcopy(self.net)
            logits = net(x_mix_s_randconv)
            loss = F.cross_entropy(logits, y1)

            grad = torch.autograd.grad(loss, net.parameters())

            total_grad.append(
                grad[grad_layer].flatten().cpu().numpy())

            total_label.append('mlti_with_randconv_grad')

        return total_grad, total_label
