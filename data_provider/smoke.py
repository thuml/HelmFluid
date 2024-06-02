import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

class SmokeDatasetMemory(Dataset):
    def __init__(self, args, split='train', var=None):
        self.data_path = args.data_path
        self.h, self.w, self.z = args.h, args.w, args.z
        self.data_files = sorted(os.listdir(Path(self.data_path)))
        self.h_down, self.w_down, self.z_down = args.h_down, args.w_down, args.z_down
        self.batch_size = args.batch_size if split == 'train' else 4
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain
        var_dict = {'f':0, 'u':1, 'v':2, 'w':3}
        if var is not None and var not in var_dict.keys():
            raise Exception('var must be None or one of [\'f\', \'u\', \'v\', \'w\']')
        elif var is not None:
            self.var = var
            self.var_idx = var_dict[var]
        else:
            self.var = None
            self.var_idx = None

        self.split = split
        if split == 'train':
            self.length = args.ntrain
        elif split == 'test':
            self.length = args.ntest
        else:
            self.length = args.ntotal
        
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        if self.var_idx is None:
            dataset_npy_path = Path(self.data_path) / f'smoke_{self.h}_{self.w}_{self.z}.npy'
        else:
            dataset_npy_path = Path(self.data_path) / f'smoke_{self.h}_{self.w}_{self.z}_{self.var}.npy'
        
        if os.path.exists(dataset_npy_path):
            dataset = np.load(dataset_npy_path)
        else:
            data_list = []
            for i in range(1200):
                data_file = Path(self.data_path) / f'smoke_{i}.npz'
                data_instance = np.load(data_file)
                data_instance = torch.cat([torch.from_numpy(data_instance['fluid_field']), torch.from_numpy(data_instance['velocity'])], dim=-1)  # 20 32 32 32 1+3=4
                data_instance = data_instance[:, :self.h:self.h_down, :self.w:self.w_down, :self.z:self.z_down]
                data_list.append(data_instance if self.var is None else data_instance[...,self.var_idx:self.var_idx+1])
            dataset = np.stack(data_list, axis=0) # n t z h w v
            np.save(str(dataset_npy_path.absolute()), dataset)
        dataset = torch.from_numpy(dataset.astype(np.float32))
        return dataset

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.split == 'train':
            idx = index
        elif self.split == 'test':
            idx = self.ntrain + index
        else:
            raise Exception('split must be train or test')

        input_x = self.dataset[idx, :self.T_in]
        input_x = rearrange(input_x, 't z h w v -> z h w v t')
        input_y = self.dataset[idx, -self.T_out:]
        input_y = rearrange(input_y, 't z h w v -> z h w v t')
        return input_x, input_y # B Z H W T*V
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False)