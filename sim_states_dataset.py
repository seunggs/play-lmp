from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class PlayLMPSimStatesDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int, O_features: int, a_features: int, transform=None):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.O_features = O_features
        self.a_features = a_features
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        sample: {
            O_seq: (seq_len, O_size),
            a_seq: (seq_len, a_size),
        }
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_idx = idx # NOTE: NOT idx * self.seq_len
        end_idx = start_idx + self.seq_len
        sample = self.df.iloc[start_idx:end_idx,:]
        sample = {
            'O_seq': sample.iloc[:,:self.O_features],
            'a_seq': sample.iloc[:,-self.a_features:],
        }
        
        if self.transform:
            sample = self.transform(sample)

        return sample
