import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import ipdb


class DermNet(Dataset):

    def __init__(self, args, mode, transform=None):
        super(DermNet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes
        self.k_shot = args.update_batch_size
        self.k_query = args.update_batch_size_eval
        self.set_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        self.data_file = '{}/DermNet/Dermnet_all_84.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))
        num_data = [(eachkey, self.data[eachkey].shape[0])
                    for eachkey in self.data]

        num_data = sorted(num_data, key=lambda x: x[1], reverse=True)

        if mode == 'train':
            sel_class_num = int(self.args.ratio*150)
            self.used_diseases = [eachid[0]
                                  for eachid in num_data[:sel_class_num]]
        elif mode == 'test':
            self.used_diseases = [eachid[0] for eachid in num_data[150:]]

        self.transform = transform
        if self.transform is None:
            for eachkey in self.data.keys():
                self.data[eachkey] = torch.tensor(np.transpose(
                    self.data[eachkey] / np.float32(255), (0, 3, 1, 2)))
        else:
            for eachkey in self.data.keys():
                self.data[eachkey] = torch.tensor(np.transpose(
                    self.data[eachkey], (0, 3, 1, 2)))

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.array(self.used_diseases)

        if self.args.train:
            support_x = torch.FloatTensor(torch.zeros(
                (self.args.meta_batch_size, self.set_size, 3, 84, 84)))
            query_x = torch.FloatTensor(torch.zeros(
                (self.args.meta_batch_size, self.query_size, 3, 84, 84)))

            support_y = np.zeros([self.args.meta_batch_size, self.set_size])
            query_y = np.zeros([self.args.meta_batch_size, self.query_size])

            for meta_batch_id in range(self.args.meta_batch_size):
                self.choose_classes = np.random.choice(
                    self.classes_idx, size=self.nb_classes, replace=False)
                for j in range(self.nb_classes):
                    self.samples_idx = np.arange(
                        self.data[self.choose_classes[j]].shape[0])
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]

                    if self.transform is None:
                        support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    :self.k_shot], ...]
                        query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    self.k_shot:], ...]
                    else:
                        support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.transform(self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    :self.k_shot], ...]) / 255.0
                        query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.transform(self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    self.k_shot:], ...]) / 255.0
                    support_y[meta_batch_id][j *
                                             self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j *
                                           self.k_query:(j + 1) * self.k_query] = j

        else:
            support_x = torch.FloatTensor(
                torch.zeros((self.set_size, 3, 84, 84)))
            support_y = np.zeros([self.set_size])
            self.choose_classes = np.random.choice(
                self.classes_idx, size=self.nb_classes, replace=False)
            query_size_test = sum([self.data[self.choose_classes[j]].shape[0]
                                  for j in range(self.nb_classes)]) - self.set_size
            query_x = torch.FloatTensor(
                torch.zeros((query_size_test, 3, 84, 84)))
            query_y = np.zeros([query_size_test])

            split_loc_pre = [self.data[self.choose_classes[j]
                                       ].shape[0]-self.k_shot for j in range(self.nb_classes)]

            query_split_loc_list = [sum(split_loc_pre[:j])
                                    for j in range(self.nb_classes+1)]

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(
                    self.data[self.choose_classes[j]].shape[0])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx
                # idx1 = idx[0:self.k_shot + self.k_query]
                if self.transform is None:
                    support_x[j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                :self.k_shot], ...]
                    query_x[query_split_loc_list[j]:query_split_loc_list[j+1]] = self.data[self.choose_classes[j]][
                        choose_samples[self.k_shot:], ...]
                else:
                    support_x[j * self.k_shot:(j + 1) * self.k_shot] = self.transform(self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                :self.k_shot], ...]) / 255.0
                    query_x[query_split_loc_list[j]:query_split_loc_list[j+1]] = self.transform(self.data[self.choose_classes[j]][
                        choose_samples[self.k_shot:], ...]) / 255.0
                support_y[j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[query_split_loc_list[j]:query_split_loc_list[j+1]] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)


class ISIC(Dataset):

    def __init__(self, args, mode, transform=None):
        super(ISIC, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes
        self.k_shot = args.update_batch_size
        self.k_query = args.update_batch_size_eval
        self.set_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        if mode == 'train':
            self.data_file = '{}/ISIC/ISIC_train.pkl'.format(args.datadir)
        elif mode == 'test':
            self.data_file = '{}/ISIC/ISIC_test.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.transform = transform
        if self.transform is None:
            for eachkey in self.data.keys():
                self.data[eachkey] = torch.tensor(np.transpose(
                    self.data[eachkey] / np.float32(255), (0, 3, 1, 2)))
        else:
            for eachkey in self.data.keys():
                self.data[eachkey] = torch.tensor(np.transpose(
                    self.data[eachkey], (0, 3, 1, 2)))

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.array(list(self.data.keys()))

        if self.args.train:
            support_x = torch.FloatTensor(torch.zeros(
                (self.args.meta_batch_size, self.set_size, 3, 84, 84)))
            query_x = torch.FloatTensor(torch.zeros(
                (self.args.meta_batch_size, self.query_size, 3, 84, 84)))

            support_y = np.zeros([self.args.meta_batch_size, self.set_size])
            query_y = np.zeros([self.args.meta_batch_size, self.query_size])

            for meta_batch_id in range(self.args.meta_batch_size):
                self.choose_classes = np.random.choice(
                    self.classes_idx, size=self.nb_classes, replace=False)
                for j in range(self.nb_classes):
                    self.samples_idx = np.arange(
                        self.data[self.choose_classes[j]].shape[0])
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    # idx1 = idx[0:self.k_shot + self.k_query]

                    if self.transform is None:
                        support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    :self.k_shot], ...]
                        query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    self.k_shot:], ...]
                    else:
                        support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.transform(self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    :self.k_shot], ...]) / 255.0
                        query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.transform(self.data[
                            self.choose_classes[
                                j]][choose_samples[
                                    self.k_shot:], ...]) / 255.0
                    support_y[meta_batch_id][j *
                                             self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j *
                                           self.k_query:(j + 1) * self.k_query] = j

        else:
            support_x = torch.FloatTensor(
                torch.zeros((self.set_size, 3, 84, 84)))
            support_y = np.zeros([self.set_size])
            self.choose_classes = np.random.choice(
                self.classes_idx, size=self.nb_classes, replace=False)
            query_size_test = self.data[self.choose_classes[0]].shape[0] + self.data[self.choose_classes[1]].shape[
                0] - self.set_size
            # print(query_size_test, self.data[self.choose_classes[0]].shape[0], self.data[self.choose_classes[1]].shape[0])
            query_x = torch.FloatTensor(
                torch.zeros((query_size_test, 3, 84, 84)))
            query_y = np.zeros([query_size_test])

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(
                    self.data[self.choose_classes[j]].shape[0])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx
                # idx1 = idx[0:self.k_shot + self.k_query]

                if self.transform is None:
                    support_x[j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                :self.k_shot], ...]
                else:
                    support_x[j * self.k_shot:(j + 1) * self.k_shot] = self.transform(self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                :self.k_shot], ...]) / 255.0
                support_y[j * self.k_shot:(j + 1) * self.k_shot] = j

                query_split_loc = self.data[self.choose_classes[0]
                                            ].shape[0]-self.k_shot

                if j == 0:
                    if self.transform is None:
                        query_x[:query_split_loc] = self.data[self.choose_classes[0]][
                            choose_samples[self.k_shot:], ...]
                    else:
                        query_x[:query_split_loc] = self.transform(self.data[self.choose_classes[0]][
                            choose_samples[self.k_shot:], ...]) / 255.0
                    query_y[:query_split_loc] = j
                else:
                    if self.transform is None:
                        query_x[query_split_loc:] = self.data[self.choose_classes[1]][
                            choose_samples[self.k_shot:], ...]
                    else:
                        query_x[query_split_loc:] = self.transform(self.data[self.choose_classes[1]][
                            choose_samples[self.k_shot:], ...]) / 255.0
                    query_y[query_split_loc:] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)


class MiniImagenet(Dataset):

    def __init__(self, args, mode, transform=None):
        super(MiniImagenet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        # number of samples per set for evaluation
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(
                args.datadir)
        elif mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(
                args.datadir)
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(
                args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.all_train_classes = np.array([26, 59, 14, 16, 17, 52,  8, 39, 46, 32, 20, 57, 34, 25, 63, 31, 30,
                                           40,  0, 43,  7, 33, 12,  6, 22, 23, 49, 50, 15, 13, 51, 10, 24, 27,
                                           47, 55,  9,  5, 18, 36, 44, 35,  4, 21, 61, 42, 11,  3, 45, 58, 60,
                                           56,  1, 28, 48, 54, 37, 19, 62, 41, 38,  2, 53, 29])
        self.num_train_use_class = int(64*self.args.ratio)

        self.transform = transform
        if self.transform is None:
            self.data = torch.tensor(np.transpose(
                self.data / np.float32(255), (0, 1, 4, 2, 3)))
        else:
            self.data = torch.tensor(np.transpose(
                self.data, (0, 1, 4, 2, 3)))

    def __len__(self):
        return self.args.metatrain_iterations*self.args.meta_batch_size

    def __getitem__(self, index):
        if self.mode == 'train':
            self.classes_idx = self.all_train_classes[:self.num_train_use_class]
        else:
            self.classes_idx = np.arange(self.data.shape[0])

        # ipdb.set_trace()
        self.samples_idx = np.arange(self.data.shape[1])

        support_x = torch.FloatTensor(torch.zeros(
            (self.args.meta_batch_size, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros(
            (self.args.meta_batch_size, self.query_size, 3, 84, 84)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_classes = np.random.choice(
                self.classes_idx, size=self.nb_classes, replace=False)
            for j in range(self.nb_classes):
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                if self.transform is None:
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]
                else:
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.transform(self.data[
                        self.choose_classes[
                            j], choose_samples[
                                :self.k_shot], ...]) / 255.0
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.transform(self.data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]) / 255.0
                support_y[meta_batch_id][j *
                                         self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j *
                                       self.k_query:(j + 1) * self.k_query] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)


class RainbowMNIST(Dataset):

    def __init__(self, args, mode, transform=None):
        super(RainbowMNIST, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes  # 10
        self.nb_samples_per_class = args.update_batch_size + \
            args.update_batch_size_eval  # 1 + 15
        self.n_way = args.num_classes  # n-way, 10
        self.k_shot = args.update_batch_size  # k-shot, 1
        self.k_query = args.update_batch_size_eval  # for evaluation, 15
        self.set_size = self.n_way * self.k_shot  # num of samples per set, 10
        # number of samples per set for evaluation, 150
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        self.data_file = '{}RainbowMNIST/rainbowmnist_all.pkl'.format(
            args.datadir)

        # data는 dict로 되어있고, dict의 key는 task 번호로 되어 있고,
        # 하나의 task는 images, labels 키로 되어 있고
        # image는 [1000, 28, 28, 3]으로 되어 있고, label은 [1000, 1]로 되어 있음. class당 sample 100개씩
        self.data = pickle.load(open(self.data_file, 'rb'))

        # task의 수, 총 56개 task
        self.num_groupid = len(self.data.keys())

        self.transform = transform
        for group_id in range(self.num_groupid):
            self.data[group_id]['labels'] = self.data[group_id]['labels'].reshape(
                10, 100)
            self.data[group_id]['images'] = self.data[group_id]['images'].reshape(
                10, 100, 28, 28, 3)
            # num class x num sample x 3 x 28 x 28
            if self.transform is None:
                self.data[group_id]['images'] = torch.tensor(
                    np.transpose(self.data[group_id]['images'], (0, 1, 4, 2, 3)))
            else:
                self.data[group_id]['images'] = torch.tensor(
                    np.transpose(self.data[group_id]['images'], (0, 1, 4, 2, 3)) * 255.0, dtype=torch.uint8)

        if self.mode == 'train':
            self.sel_group_id = np.array([49,  8, 19, 47, 25, 27, 42, 50, 24, 40,  3, 45,  6, 41,  2, 17, 14,
                                          10,  5, 26, 12, 33,  9, 11, 32, 54, 28,  7, 39, 51, 46, 44, 30, 13,
                                          18,  0, 34, 43, 52, 29])
            num_of_tasks = self.sel_group_id.shape[0]
            if self.args.ratio < 1.0:
                # 논문대로 16개임
                num_of_tasks = int(num_of_tasks*self.args.ratio)
                self.sel_group_id = self.sel_group_id[:num_of_tasks]
        elif self.mode == 'val':
            self.sel_group_id = np.array([15, 16, 38, 36, 37,  4])
        elif self.mode == 'test':
            self.sel_group_id = np.array(
                [35, 48, 23, 20, 22, 55,  1, 21, 31, 53])

    def __len__(self):
        return self.args.metatrain_iterations*self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.arange(self.data[0]['images'].shape[0])  # 0~9
        self.samples_idx = np.arange(self.data[0]['images'].shape[1])  # 0-99

        # 4 x 10(10-way 1 shot) x 3 x 28 x 28
        support_x = torch.FloatTensor(torch.zeros(
            (self.args.meta_batch_size, self.set_size, 3, 28, 28)))
        # 4 x 150(10-way 15 shot) x 3 x 28 x 28
        query_x = torch.FloatTensor(torch.zeros(
            (self.args.meta_batch_size, self.query_size, 3, 28, 28)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_group = np.random.choice(
                self.sel_group_id, size=1, replace=False).item()  # random으로 task 선택
            for j in range(10):
                np.random.shuffle(self.samples_idx)  # 0-99 shuffle
                # support와 query 다 뽑음(1+15)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                # j class에 k shot만큼 support, 나머지는 query
                # (class x sample) 순임
                if self.transform is None:
                    support_x[meta_batch_id][j * self.k_shot:(
                        j + 1) * self.k_shot] = self.data[self.choose_group]['images'][j, choose_samples[:self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[self.choose_group]['images'][j, choose_samples[
                        self.k_shot:], ...]
                else:
                    support_x[meta_batch_id][j * self.k_shot:(
                        j + 1) * self.k_shot] = self.transform(self.data[self.choose_group]['images'][j, choose_samples[:self.k_shot], ...]) / 255.0
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.transform(self.data[self.choose_group]['images'][j, choose_samples[
                        self.k_shot:], ...]) / 255.0
                support_y[meta_batch_id][j *
                                         self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j *
                                       self.k_query:(j + 1) * self.k_query] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)


class Metabolism(Dataset):

    def __init__(self, args, mode):
        super(Metabolism, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        # number of samples per set for evaluation
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        self.data_file = '{}/metabolism_data_new.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.num_groupid = len(self.data.keys())

        self.index_label = {}

        self.all_group_list = ['CYP1A2_Veith', 'CYP3A4_Veith', 'CYP2D6_Veith', 'CYP2C9_Substrate_CarbonMangels',
                               'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'CYP2C19_Veith',
                               'CYP2C9_Veith']

        if self.mode == 'train':
            self.sel_group_id = np.array(['CYP1A2_Veith', 'CYP3A4_Veith', 'CYP2D6_Veith', 'CYP2C9_Substrate_CarbonMangels',
                                          'CYP2D6_Substrate_CarbonMangels'])
        elif self.mode == 'val':
            self.sel_group_id = np.array(
                ['CYP3A4_Substrate_CarbonMangels', 'CYP2C19_Veith', 'CYP2C9_Veith'])
        elif self.mode == 'test':
            self.sel_group_id = np.array(
                ['CYP3A4_Substrate_CarbonMangels', 'CYP2C19_Veith', 'CYP2C9_Veith'])

        for group_id in self.all_group_list:
            self.data[group_id]['label'] = self.data[group_id]['label']
            self.data[group_id]['feature'] = self.data[group_id]['feature']

            self.index_label[group_id] = {}
            if 'Substrate' not in group_id:
                self.index_label[group_id][0] = np.nonzero(
                    self.data[group_id]['label'] == 0.0)[0]
                self.index_label[group_id][1] = np.nonzero(
                    self.data[group_id]['label'] == 1.0)[0]
            else:
                self.index_label[group_id][0] = np.nonzero(
                    self.data[group_id]['label'] == 0.0)[0]
                self.index_label[group_id][1] = np.nonzero(
                    self.data[group_id]['label'] == 1.0)[0]

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = 2
        support_x = np.zeros((self.args.meta_batch_size, self.set_size, 1024))
        query_x = np.zeros((self.args.meta_batch_size, self.query_size, 1024))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_group = np.random.choice(
                self.sel_group_id, size=1, replace=False).item()
            for j in range(2):
                self.samples_idx = np.array(
                    self.index_label[self.choose_group][j])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = \
                    self.data[self.choose_group]['feature'][choose_samples[:self.k_shot], ...].astype(
                        float)
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = \
                    self.data[self.choose_group]['feature'][choose_samples[self.k_shot:], ...].astype(
                        float)

                support_y[meta_batch_id][j *
                                         self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j *
                                       self.k_query:(j + 1) * self.k_query] = j

        return torch.FloatTensor(support_x), torch.LongTensor(support_y), torch.FloatTensor(query_x), torch.LongTensor(
            query_y)


class NCI(Dataset):

    def __init__(self, args, mode):
        super(NCI, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        # number of samples per set for evaluation
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        self.data_file = '{}/NCI_data_new.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.num_groupid = len(self.data.keys())

        self.index_label = {}

        self.all_group_list = [81, 41, 83, 47, 109, 145, 33, 1, 123]

        if self.mode == 'train':
            self.sel_group_id = np.array([81, 41, 83, 47, 109, 145])
            assert self.args.ratio == 1.0
        elif self.mode == 'val':
            self.sel_group_id = np.array([33, 1, 123])
        elif self.mode == 'test':
            self.sel_group_id = np.array([33, 1, 123])

        for group_id in self.all_group_list:
            self.data[group_id]['label'] = self.data[group_id]['label']
            self.data[group_id]['feature'] = self.data[group_id]['feature']

            self.index_label[group_id] = {}
            if group_id in [81, 41, 83, 47, 109, 145]:
                self.index_label[group_id][0] = np.nonzero(
                    self.data[group_id]['label'] == -1.0)[0][:500]
                self.index_label[group_id][1] = np.nonzero(
                    self.data[group_id]['label'] == 1.0)[0][:500]
            else:
                self.index_label[group_id][0] = np.nonzero(
                    self.data[group_id]['label'] == -1.0)[0]
                self.index_label[group_id][1] = np.nonzero(
                    self.data[group_id]['label'] == 1.0)[0]

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = 2
        support_x = np.zeros((self.args.meta_batch_size, self.set_size, 1024))
        query_x = np.zeros((self.args.meta_batch_size, self.query_size, 1024))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_group = np.random.choice(
                self.sel_group_id, size=1, replace=False).item()
            for j in range(2):
                self.samples_idx = np.array(
                    self.index_label[self.choose_group][j])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = \
                    self.data[self.choose_group]['feature'][choose_samples[:self.k_shot], ...].astype(
                        float)
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = \
                    self.data[self.choose_group]['feature'][choose_samples[self.k_shot:], ...].astype(
                        float)

                support_y[meta_batch_id][j *
                                         self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j *
                                       self.k_query:(j + 1) * self.k_query] = j

        return torch.FloatTensor(support_x), torch.LongTensor(support_y), torch.FloatTensor(query_x), torch.LongTensor(query_y)
