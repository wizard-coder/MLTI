import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist
import numpy as np
from torch.distributions import Beta
from rand_conv import RandConvModule


class Protonet(nn.Module):
    def __init__(self, args, learner, rand_conv=False, rand_conv_prob=0.0, rand_conv_mixing=False):
        super(Protonet, self).__init__()
        self.args = args
        self.learner = learner
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.device = args.device

        self.rand_conv = rand_conv
        self.rand_conv_prob = rand_conv_prob
        self.rand_conv_mixing = rand_conv_mixing

        print(self.rand_conv)

    def forward(self, xs, ys, xq, yq):

        if self.rand_conv and self.args.train:
            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob).to(self.device)

            xs = rand_conv(xs).detach()
            xq = rand_conv(xq).detach()

        x = torch.cat([xs, xq], 0)

        z = self.learner(x)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(1)
        zq = z[self.args.num_classes * self.args.update_batch_size:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = []
        for i in range(self.args.num_classes * self.args.update_batch_size_eval):
            loss_val.append(-log_p_y[i, yq[i]])

        loss_val = torch.stack(loss_val).squeeze().mean()

        _, y_hat = log_p_y.max(1)

        acc_val = torch.eq(y_hat, yq).float().mean()

        return loss_val, acc_val

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

    def mixup_data(self, xs, xq, lam):
        mixed_x = xq.clone()
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(xq.size(), lam)

        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = xs[:, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (xq.size()[-1] * xq.size()[-2]))

        return mixed_x, lam

    def forward_crossmix(self, x1s, y1s, x1q, y1q, x2s, y2s, x2q, y2q):
        lam_mix = self.dist.sample().to(self.device)

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

        x_mix_s, _ = self.mixup_data(x1s, x2s, lam_mix)

        x_mix_q, _ = self.mixup_data(x1q, x2q, lam_mix)

        if self.rand_conv:
            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob).to(self.device)

            x_mix_s = rand_conv(x_mix_s).detach()
            x_mix_q = rand_conv(x_mix_q).detach()

        x = torch.cat([x_mix_s, x_mix_q], 0)

        z = self.learner(x)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(
            1)
        zq = z[self.args.num_classes * self.args.update_batch_size:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = []
        for i in range(self.args.num_classes * self.args.update_batch_size_eval):
            loss_val.append(-log_p_y[i, y1q[i]])

        loss_val = torch.stack(loss_val).squeeze().mean()

        _, y_hat = log_p_y.max(1)

        acc_val = torch.eq(y_hat, y1q).float().mean()

        return loss_val, acc_val

    def get_proto(self, x1, x2, num_samples=100):
        # proto 분포 분석용 코드

        total_proto = []
        total_label = []

        z = self.learner(x1)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(1)[0].flatten().cpu().detach().numpy()

        total_proto.append(z_proto)
        total_label.append('x1_proto')

        z = self.learner(x2)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(1)[0].flatten().cpu().detach().numpy()

        total_proto.append(z_proto)
        total_label.append('x2_proto')

        # rand conv
        for _ in range(num_samples):

            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob).to(self.device)

            x1_rand_conv = rand_conv(x1).detach()
            x2_rand_conv = rand_conv(x2).detach()

            z = self.learner(x1_rand_conv)

            z_dim = z.size(-1)

            z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                                   self.args.update_batch_size, z_dim).mean(1)[0].flatten().cpu().detach().numpy()

            total_proto.append(z_proto)
            total_label.append('x1_randconv_proto')

            z = self.learner(x2_rand_conv)

            z_dim = z.size(-1)

            z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                                   self.args.update_batch_size, z_dim).mean(1)[0].flatten().cpu().detach().numpy()

            total_proto.append(z_proto)
            total_label.append('x2_randconv_proto')

        # mlti

        for _ in range(num_samples):
            lam_mix = self.dist.sample().to(self.device)

            task_2_shuffle_id = np.arange(self.args.num_classes)
            np.random.shuffle(task_2_shuffle_id)
            task_2_shuffle_id_s = np.array(
                [np.arange(self.args.update_batch_size) + task_2_shuffle_id[idx] * self.args.update_batch_size for idx in
                 range(self.args.num_classes)]).flatten()

            x2_shuffle = x2[task_2_shuffle_id_s]

            x_mix_s, _ = self.mixup_data(x1, x2_shuffle, lam_mix)

            z = self.learner(x_mix_s)

            z_dim = z.size(-1)

            z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                                   self.args.update_batch_size, z_dim).mean(1)[0].flatten().cpu().detach().numpy()

            total_proto.append(z_proto)
            total_label.append('mlti_proto')

            rand_conv = RandConvModule(kernel_size=[1, 3, 5, 7],
                                       in_channels=3,
                                       out_channels=3,
                                       mixing=self.rand_conv_mixing,
                                       identity_prob=self.rand_conv_prob).to(self.device)

            x_mix_s_rand_conv = rand_conv(x_mix_s).detach()

            z = self.learner(x_mix_s_rand_conv)

            z_dim = z.size(-1)

            z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                                   self.args.update_batch_size, z_dim).mean(1)[0].flatten().cpu().detach().numpy()

            total_proto.append(z_proto)
            total_label.append('mlti_randconv_proto')

        return total_proto, total_label
