# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
from IPython import embed
import numpy as np
import torch


class eDreamsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return torch.Tensor(self.data[idx]), torch.from_numpy(np.asarray(self.labels[idx])).type(torch.LongTensor)




