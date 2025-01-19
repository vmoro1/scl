# Dataset when having one constraint per sample (i.e. not only average constraints). For this 
# case, we need to keep track of the sample indices to know which constraints correspond to which samples.

import torch


class DatasetPerSampleConstraints(torch.utils.data.Dataset):
    """Dataset that can handle problems with per sample 
    constraints (and not only average constraints)"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.samples_indices = torch.arange(self.x.shape[0])
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.samples_indices[idx], self.x[idx], self.y[idx]
    