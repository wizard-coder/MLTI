import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from torchvision import transforms


class RainbowMNIST(Dataset):

    def __init__(self, args, mode, transform=None, extreme_dist=None):
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
            if extreme_dist is None:
                self.sel_group_id = np.array([49,  8, 19, 47, 25, 27, 42, 50, 24, 40,  3, 45,  6, 41,  2, 17, 14,
                                              10,  5, 26, 12, 33,  9, 11, 32, 54, 28,  7, 39, 51, 46, 44, 30, 13,
                                              18,  0, 34, 43, 52, 29])
                num_of_tasks = self.sel_group_id.shape[0]
                if self.args.ratio < 1.0:
                    # 논문대로 16개임
                    num_of_tasks = int(num_of_tasks*self.args.ratio)
                    self.sel_group_id = self.sel_group_id[:num_of_tasks]
            else:
                print("extream dist")
                # 6개 task씩 할당
                # color는 red, indigo, blue, orange, green, violet
                if extreme_dist == "color":
                    # violet
                    # 3개 full, 3개 half
                    # 0도 1개, 90도 1개, 180도 1개, 270도 2개
                    # (violet, full, 270) : 3
                    # (violet, half, 180) : 6
                    # (violet, full, 180) : 2
                    # (violet, half, 90) : 5
                    # (violet, half, 270) : 7
                    # (violet, full, 0) : 0
                    self.sel_group_id = np.array([3, 6, 2, 5, 7, 0])
                # scale full/half
                elif extreme_dist == "scale":
                    # 모든 색상
                    # full
                    # 0 1개, 90 2개, 180 1개, 270 2개
                    # (red, full, 90) : 49
                    # (indigo, full, 0): 8
                    # (blue, full, 270): 19
                    # (orange, full, 180): 42
                    # (green, full, 90): 25
                    # (violet, full, 270): 3
                    self.sel_group_id = np.array([49, 8, 19, 42, 25, 3])
                # rotation 0, 90, 180, 270
                elif extreme_dist == "rotation":
                    # 모든 색상(blue는 없어서 다른색상으로)
                    # full 3, half 3
                    # 0도
                    # (red, half, 0): 52
                    # (indigo, full, 0): 8
                    # (blue, full, 0): 16
                    # (orange, half, 0): 44
                    # (green, full, 0): 24
                    # (violet, half, 0): 4
                    self.sel_group_id = np.array([52, 8, 16, 44, 24, 4])
                elif extreme_dist == "compare":
                    # all color
                    # 3개 full, 3개 half
                    # 0도 1개, 90도 2개, 180도 1개, 270도 2개
                    # (red, full, 90) : 49
                    # (indigo, half, 90): 13
                    # (blue, full, 0): 16
                    # (orange, half, 0): 44
                    # (green, half, 180): 30
                    # (violet, full, 270): 3
                    self.sel_group_id = np.array([49, 13, 16, 44, 30, 3])
                else:
                    raise NotImplementedError

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
