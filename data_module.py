from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from sim_states_dataset import PlayLMPSimStatesDataset
import utils.functions as CF


class PlayLMPSimStatesDataModule(pl.LightningDataModule):
    def __init__(self, data_root_path: str, seq_len: int, O_features: int, a_features: int, bs: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_path = (Path(data_root_path) / 'dataset.csv').as_posix()

        # compute mean and std for normalization (for training)
        df = pd.read_csv(self.train_data_path)
        O_df = df.iloc[:,:O_features]
        a_df = df.iloc[:,-a_features:]
        O_mean = O_df.mean(axis=0)
        O_std = O_df.std(axis=0)
        a_mean = a_df.mean(axis=0)
        a_std = a_df.std(axis=0)

        self.seq_len = seq_len
        self.O_features = O_features
        self.a_features = a_features

        self.num_workers = num_workers # set it to # of cpu's in the machine
        self.bs = bs # effective batch size of 2048 / (8 GPUs/instance * 4 instances)
        self.transform = transforms.Compose([CF.ToTensor(), CF.Normalize(O_mean, O_std, a_mean, a_std)])
        self.val_ratio = 0.2

    def setup(self, stage=None):
        # called on every GPU
        if stage == 'fit' or stage is None:     
            self.x_full = PlayLMPSimStatesDataset(
                self.data_path, seq_len=self.seq_len, O_features=self.O_features, a_features=self.a_features, transform=self.transform)
            
            # cannot use torch.random_split since data is continuous
            x_full_len = len(self.x_full)
            train_end_idx = int(x_full_len * (1 - self.val_ratio))
            self.x_train = self.x_full[:train_end_idx]
            self.x_val = self.x_full[train_end_idx:]
        
        # self.dims = tuple(self.x_train[0]['O_seq'].shape)

    def train_dataloader(self):
        return DataLoader(self.x_train, batch_size=self.bs, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.x_val, batch_size=self.bs, shuffle=True, num_workers=self.num_workers, pin_memory=True)