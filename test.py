import torch
from torchvision import transforms
import pickle
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str, choices=['test1', 'test2'])

args = parser.parse_args()

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


data_file = '/home/rsh/research/data/MLTI/RainbowMNIST/rainbowmnist_all.pkl'

data = pickle.load(open(data_file, 'rb'))

for group_id in range(len(data.keys())):
    data[group_id]['images'] = data[group_id]['images'].reshape(
        10, 100, 28, 28, 3)

    data[group_id]['images'] = torch.tensor(
        np.transpose(data[group_id]['images'], (0, 1, 4, 2, 3))*255.0, dtype=torch.uint8)


print(data[0]['images'][0][0].shape)
print(data[0]['images'][0][0].dtype)
print(torch.max(data[0]['images'][0][0]))


# composed = transforms.Compose([transforms.TrivialAugmentWide()])


# sample = data[0][0]

# transformed_sample = composed(sample)
# print((transformed_sample/255.0).dtype)

# img_list = [sample, transformed_sample]

# grid = make_grid(img_list, 2)
# show(grid)
# plt.show()
