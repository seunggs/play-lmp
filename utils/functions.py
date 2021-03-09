import torch


class ToTensor():
    '''
    Convert pd dataframe to tensors
    '''
    def __call__(self, sample_df):
        O_seq, a_seq = sample_df['O_seq'], sample_df['a_seq']

        return {
            'O_seq': torch.tensor(O_seq.values, device=self.device),
            'a_seq': torch.tensor(a_seq.values, device=self.device)
        }


class Normalize():
    '''
    Normalize a tensor along each step in the given sample sequence
    mean and std over all samples (row) per feature (col)
    '''
    def __init__(self, O_mean, O_std, a_mean, a_std):
        self.O_mean = torch.tensor(O_mean, device=self.device)[None,:]
        self.O_std = torch.tensor(O_std, device=self.device)[None,:]
        self.a_mean = torch.tensor(a_mean, device=self.device)[None,:]
        self.a_std = torch.tensor(a_std, device=self.device)[None,:]

    def __call__(self, sample):
        O_seq, a_seq = sample['O_seq'], sample['a_seq']

        return {
            'O_seq': (O_seq - self.O_mean) / self.O_std,
            'a_seq': (a_seq - self.a_mean) / self.a_std
        }


class Unnormalize():
    '''
    Unnormalize a tensor along each step in the given sample sequence
    '''
    def __init__(self, O_mean, O_std, a_mean, a_std):
        self.O_mean = torch.tensor(O_mean, device=self.device)[None,:]
        self.O_std = torch.tensor(O_std, device=self.device)[None,:]
        self.a_mean = torch.tensor(a_mean, device=self.device)[None,:]
        self.a_std = torch.tensor(a_std, device=self.device)[None,:]

    def __call__(self, sample):
        O_seq, a_seq = sample['O_seq'], sample['a_seq']

        return {
            'O_seq': (O_seq * self.O_std) + self.O_mean,
            'a_seq': (a_seq * self.a_std) + self.a_mean
        }


def one_hot_encoding(labels, num_classes):
    y = torch.eye(num_classes, device=self.device)
    return y[labels]

