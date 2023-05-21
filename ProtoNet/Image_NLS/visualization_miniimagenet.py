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

data = pickle.load(open(os.path.expanduser('~/research/data/PHD/lecture/AI602/MLTI/miniImagenet/mini_imagenet_train.pkl') , 'rb'))

data = torch.tensor(np.transpose(data, (0, 1, 4, 2, 3)))

num_images = 18
img_list = []

for i in range(num_images):
    choose_class = np.random.randint(0, 64)
    choose_sample = np.random.randint(0, 600)

    img_list.append(data[choose_class, choose_sample, :, :, :])

grid = make_grid(img_list, 6)
show(grid)
plt.show()