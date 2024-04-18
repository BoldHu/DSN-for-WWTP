import torch.utils.data as data
import os
import scipy.io as sio
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_label_root, transform=None):
        self.data_root = os.path.join("./Data", data_root)
        self.data_label_root = os.path.join("./Data", data_label_root)
        self.transform = transform
        
        # load data
        self.data = sio.loadmat(self.data_root)['data']
        self.data_label = sio.loadmat(self.data_label_root)['EQvec']
        
        # transform the data and labels: standardize
        if self.transform:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)
            # standardize the labels
            # self.data_label = scaler.fit_transform(self.data_label)
        
    def __getitem__(self, item):
        d = self.data[item]
        l = self.data_label[item]
        
        # convert to tensor
        d = torch.tensor(d, dtype=torch.float32)
        l = torch.tensor(l, dtype=torch.float32)
        
        return d, l
    
    def __len__(self):
        return len(self.data)