import pickle
import os
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import ImageDraw, Image

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
    '~/research/data/MLTI/RainbowMNIST/rainbowmnist_all.pkl'), 'rb'))


num_images = 100
img_list = []

# for i in range(num_images):
#     choose_group = np.random.randint(0, 56)
#     choose_class = np.random.randint(0, 10)
#     choose_sample = np.random.randint(0, 100)

#     np_img = data[choose_group]['images'].reshape(10, 100, 28, 28, 3)
#     np_img = np_img[choose_class, choose_sample, :, :, :]
#     np_img = np.transpose(np_img, (2, 0, 1))

#     img_list.append(torch.from_numpy(np_img))

group = [49,  8, 19, 47, 25, 27, 42, 50, 24, 40,  3, 45,  6, 41,  2, 17, 14,
         10,  5, 26, 12, 33,  9, 11, 32, 54, 28,  7, 39, 51, 46, 44, 30, 13,
         18,  0, 34, 43, 52, 29]

for i in group:
    choose_group = i
    choose_class = np.random.randint(0, 10)
    choose_sample = np.random.randint(0, 100)

    np_img = data[choose_group]['images'].reshape(10, 100, 28, 28, 3)
    np_img = np_img[choose_class, choose_sample, :, :, :]
    np_img = np.transpose(np_img, (2, 0, 1))

    tensor = torch.from_numpy(np_img)

    to_pil = torchvision.transforms.ToPILImage()

    pil_img = to_pil(tensor)

    ImageDraw.Draw(pil_img).text((0, 0), f'{i}', fill=(255, 255, 255))

    totensor = torchvision.transforms.ToTensor()

    img_list.append(totensor(pil_img))

grid = make_grid(img_list, 10)
show(grid)
plt.show()
