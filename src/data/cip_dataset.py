# dataset for counterfactual inference planning
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader


def _sample_seed(config, sample_seed=None):
    # Match GIFT's CIP evaluation protocol: every tau rebuilds CIPDataset and
    # samples history lengths from the same experiment seed.
    return int(config.exp.seed if sample_seed is None else sample_seed)


def planning_repeats(config):
    name = str(getattr(config.dataset, "name", "")).lower()
    if "mimic" in name:
        return 3
    if "cancer" in name or "tumor" in name:
        return 5
    return int(config.exp.repeats)


class CIPDataset(Dataset):
    def __init__(self, data, config, train=False, sample_seed=None):
        self.data = data
        self.train = train
        self.tau = config.exp.tau
        self.config = config
        self.is_mimic = 'mimic' in str(config.dataset.name).lower()
        if self.is_mimic and hasattr(config.dataset, 'min_seq_length'):
            # GIFT's MIMIC planning dataset samples history lengths against
            # min_seq_length, then uses the real target slice as future vitals.
            self.max_history_length = int(config.dataset.min_seq_length)
        else:
            self.max_history_length = int(config.dataset.max_seq_length) - 1
        self.min_h = int(getattr(config.dataset, 'min_history_length', 20))
        self.repeats = planning_repeats(config)
        np.random.seed(_sample_seed(config, sample_seed=sample_seed))
        history_high = self.max_history_length - self.tau
        if history_high <= self.min_h:
            raise ValueError(
                f"Invalid CIP history sampling bounds: min_history_length={self.min_h}, "
                f"max_history_length={self.max_history_length}, tau={self.tau}."
            )
        # 生成不重复的history lengths
        if train:
            # arange from 5 to max_history_length - tau
            # self.history_lengths = np.arange(1, self.max_history_length - self.tau)
            self.history_lengths = np.random.randint(self.min_h, history_high, self.repeats * 4)
            # self.history_lengths = np.arange(5, 6)
        else:
            self.history_lengths = np.random.randint(self.min_h, history_high, self.repeats)
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
            hi = max(self.max_history_length - self.tau - history_length, 1)
            start_idx = 0 if self.is_mimic else np.random.randint(0, hi)
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
        if 'sample_indices' in self.data:
            H_t['sample_indices'] = self.data['sample_indices'][data_index]
        

        # print(f'keys of H_t: {H_t.keys()}')
        # print(f'history_length: {history_length}, tau: {self.tau}') 
        # print(f"sample['outputs'].shape: {sample['outputs'].shape}")
        # print(f"self.data['outputs'][0].shape: {self.data['outputs'][0].shape}")
        target = {k: v[history_length+start_idx:history_length+self.tau+start_idx] for k, v in sample.items() if hasattr(v, '__len__')}
        if self.is_mimic:
            H_t['sequence_lengths'] = history_length
            if 'vitals' not in target:
                raise KeyError("GIFT-aligned MIMIC evaluation requires target['vitals'] for future_vitals.")
            H_t['future_vitals'] = target['vitals']

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
