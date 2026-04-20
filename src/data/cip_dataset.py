# dataset for counterfactual inference planning
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

class CIPDataset(Dataset):
    def __init__(self, data, config, train=False):
        self.data = data
        self.train = train
        self.tau = config.exp.tau
        self.max_history_length = config.dataset.max_seq_length - 1
        self.repeats = config.exp.repeats
        np.random.seed(config.exp.seed)
        # 生成不重复的history lengths
        if train:
            # arange from 5 to max_history_length - tau
            # self.history_lengths = np.arange(1, self.max_history_length - self.tau)
            self.history_lengths = np.random.randint(20, self.max_history_length - self.tau, self.repeats * 4)
            # self.history_lengths = np.arange(5, 6)
        else:
            self.history_lengths = np.random.randint(20, self.max_history_length - self.tau, self.repeats)
            # self.history_lengths = np.arange(5, 6)
        self.history_lengths = np.unique(self.history_lengths)
        self.repeats = len(self.history_lengths)
        
        # 计算每个history length对应的样本数
        self.samples_per_history = len(self.data['outputs'])
        self.model = config.model.name

    def __len__(self):
        return len(self.data['outputs']) * self.repeats

    def __getitem__(self, index):
        history_group = index // self.samples_per_history
        data_index = index % self.samples_per_history
        
        history_length = self.history_lengths[history_group]

        if not self.train:
            # print(f"self.max_history_length: {self.max_history_length}, self.tau: {self.tau}, history_length:{history_length}")
            start_idx = np.random.randint(0, self.max_history_length - self.tau - history_length)
        else:
            start_idx = 0
        # print(f"start_idx: {start_idx}")
        
        # print(f'keys: {self.data.keys()}')
        sample = {k: v[data_index] for k, v in self.data.items() 
                 if hasattr(v, '__len__') and len(v) == len(self.data['outputs'])}

        # for key in sample:
        #     print(f"sample[{key}].shape: {sample[key].shape}")

        H_t = {k: v[start_idx:history_length+start_idx] for k, v in sample.items() if hasattr(v, '__len__')}
        # append no length to the history
        for k, v in sample.items():
            if not hasattr(v, '__len__'):
                H_t[k] = v
            elif len(v) <= 2:
                H_t[k] = v

        if sample['static_features'].ndim != sample['outputs'].ndim:
            H_t['static_features'] = sample['static_features']
        

        # print(f'keys of H_t: {H_t.keys()}')
        # print(f'history_length: {history_length}, tau: {self.tau}') 
        # print(f"sample['outputs'].shape: {sample['outputs'].shape}")
        # print(f"self.data['outputs'][0].shape: {self.data['outputs'][0].shape}")
        target = {k: v[history_length+start_idx:history_length+self.tau+start_idx] for k, v in sample.items() if hasattr(v, '__len__')}

        # for key in H_t:
        #     # if key == 'static_features':
        #     print(f"H_t[{key}].shape: {H_t[key].shape}")
        
        # for key in target:
        #     # if key == 'static_features':
        #     print(f"target[{key}].shape: {target[key].shape}")
        
        return H_t, target

def get_dataloader(dataset, batch_size, shuffle=True, seed=10):
    def batch_sampler():
        # np.random.seed(seed)
        for h_idx in range(dataset.repeats):
            
            start_idx = h_idx * dataset.samples_per_history
            end_idx = (h_idx + 1) * dataset.samples_per_history
            
            indices = list(range(start_idx, end_idx))
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                yield indices[i:min(i + batch_size, len(indices))]

    return DataLoader(dataset, batch_sampler=list(batch_sampler()))
