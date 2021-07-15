import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.io as tvio

from skimage import io, transform
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

DIR = "./samples/2021_03_23_20_28_39/"
CSV_FILENAME = "data_sorted.csv"
SEED = 42

def show_image(image):
    """Show image"""
    plt.imshow(image, aspect="auto")
    plt.pause(0.001)  # pause a bit so that plots are updated

class HaloReachDataset(Dataset):
    """Halo Reach dataset."""

    def __init__(self, csv_file_name, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with filenames and controller states.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.halo_frame = pd.read_csv(os.path.join(root_dir, csv_file_name))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.halo_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                os.path.basename(self.halo_frame.iloc[idx, 0]))
        image = tvio.read_image(img_name)
        image = image / 255.0
        controller_state = self.halo_frame.iloc[idx, 1:21]
        controller_state = np.array([controller_state])
        controller_state = controller_state.astype('float').flatten()
        sample = {'image': image, 'controller_state': controller_state}

        if self.transform:
            print(self.transform)
            sample['image'] = self.transform(sample['image'])

        return sample

if __name__ == "__main__":
    print("Starting")
    file_data = pd.read_csv(os.path.join(DIR, CSV_FILENAME))
    print(file_data)
    file_names = file_data.iloc[:, 0]
    controller_states = file_data.iloc[:, 1:21]
    timestamps = file_data.iloc[:, 21]
    N = len(file_names)
    i = 11700

    img_name = os.path.basename(file_names.iloc[i])
    fig = plt.figure(dpi = 150)
    show_image(io.imread(os.path.join(DIR, img_name)))
    plt.show()