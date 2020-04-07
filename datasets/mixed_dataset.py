"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        #self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        self.dataset_list = ['3dpw','mpi-inf-3dhp']
        #self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}
        self.dataset_dict = {'3dpw': 0, 'mpi-inf-3dhp': 1}
        
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        self.partition = [.5,
                          0.5]
        self.partition = np.array(self.partition).cumsum()
        print(self.partition)

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(2): # Change range 6 to 2, since only two datasets will be used
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
