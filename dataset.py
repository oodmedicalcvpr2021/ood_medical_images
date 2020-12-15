import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
import os
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Cutout(object):
    """Randomly mask out one or more patches from an image.
       https://arxiv.org/abs/1708.04552
    Args:
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        h = img.size(1)
        w = img.size(2)

        if np.random.choice([0, 1]):
            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img


test_input_transforms = Compose([
    Resize((224, 224)),
    ToTensor()
])

train_input_transforms = Compose([
    RandomResizedCrop((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    Cutout(16 * 7),
])



def get_label(name, df, index):
    if name == 'skeletal-age':
        label = df.iloc[index, 1] - 1  # there is no 0 month label
    elif name == 'mura':
        label = df.iloc[index, 1]
    elif name == 'retina':
        label = df.iloc[index, 1]
    elif name == 'mimic-crx':
        label = df.iloc[index, 2]
    elif name == 'drimdb':
        label = df.iloc[index, 1]
    else:
        raise NotImplementedError

    return np.array(label, dtype=np.long)

def open_file(o):
    if not os.path.exists(o):
        print(o)
        raise FileNotFoundError()
    if '.npy' in o:
        return Image.fromarray(np.load(o)).convert('RGB')
    else:  # jpg
        return Image.open(str(o)).convert('RGB')


class OODDataset(Dataset):

    def __init__(self,
                 name=None,
                 mode=None,
                 root_dir=None,
                 csv_file=None,
                 load_memory=False):
        super(OODDataset, self).__init__()
        self.name = name
        self.mode = mode
        assert self.mode in ['train', 'test']
        assert self.name in ['retina', 'skeletal-age', 'mura', 'mimic-crx', 'drimdb']
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.load_memory = load_memory

        # Load into cpu ?
        if self.load_memory:
            self.image_arrs = []
            for index in range(len(self.df)):
                image_filename = f"{self.root_dir}/{self.df.iloc[index, 0]}"
                try:
                    self.image_arrs.append(open_file(image_filename))
                except:
                    print(f"image_filename: {image_filename}")
                    raise Exception

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        if self.load_memory:
            image_arr = self.image_arrs[index]
        else:
            image_arr = open_file(f"{self.root_dir}/{self.df.iloc[index, 0]}")

        if self.mode == "train":
            inputs = train_input_transforms(image_arr)
        elif self.mode == "test":
            inputs = test_input_transforms(image_arr)
        else:
            raise NotImplementedError

        label = get_label(self.name, self.df, index)

        return inputs, label
