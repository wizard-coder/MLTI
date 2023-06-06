import pickle
import os
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch

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


data = pickle.load(open(os.path.expanduser(
    '~/research/data/MLTI/DermNet/Dermnet_all_84.pkl'), 'rb'))

for eachkey in data.keys():
    data[eachkey] = torch.tensor(np.transpose(data[eachkey], (0, 3, 1, 2)))

num_images = 40
img_list = []
key_list = [key for key in data.keys()]

for i in range(num_images):

    choose_classes = np.random.choice(key_list, 1, replace=False)
    choose_sample = np.random.randint(0, data[choose_classes[0]].shape[0])

    img_list.append(data[choose_classes[0]][choose_sample, ...])

grid = make_grid(img_list, 10)
show(grid)
plt.show()
