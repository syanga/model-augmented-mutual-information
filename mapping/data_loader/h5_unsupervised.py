from torchvision import datasets, transforms
from ..base import BaseDataLoader
from torch.utils import data
import h5py
import numpy as np
import torch


class UnsupervisedDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, 
                 validation_split, test_split, num_workers, training=True, seed=0):
        self.data_dir = data_dir
        self.seed = seed
        self.dataset = UnsupervisedDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers, seed=seed)


class UnsupervisedDataset(data.Dataset):
    """ h5 file; no labels """
    def __init__(self, h5_path):
        df = h5py.File(h5_path, "r")
        xh = df.get("X")
        self.length = xh.shape[0]

        self.X = np.empty(xh.shape, order="C")
        xh.read_direct(self.X, None, None)
        df.close()

        self.X = torch.from_numpy(
            np.ascontiguousarray(self.X)).type(torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.X[index]
