import os

from torch.utils.data import Dataset
import numpy as np
import pickle

class InputHandle(Dataset):
    def __init__(self, input_param):
        self.path = input_param['path']
        self.file_path = input_param['file_path']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.length = input_param['total_length']
        self.input_length = input_param['input_length']
        self.type = input_param['type'] #train/test/valid

        if self.file_path == 'real_1':
            self.train_length = 100
            self.test_length = 50
        elif self.file_path == 'real_2':
            self.train_length = 126
            self.test_length = 63
        elif self.file_path == 'real_3':
            self.train_length = 93
            self.test_length = 46

        self.images = np.load(os.path.join(self.path, self.file_path, 'imgs.npy'))
        self.sdf = np.load(os.path.join(self.path, self.file_path, 'sdf.npy'))
        self.mask = self.sdf < -2
        self.boundary = (self.sdf<-2)*1.0*(self.sdf > -3)
        self.boundary = self.boundary == 1

    def __len__(self):
        if self.type == 'train':
            return self.train_length - self.length
        else:
            return 1

    def __getitem__(self, index):
        if self.type == 'train':
            data = self.images[index:index+self.length]
        else:
            data = self.images[self.train_length-self.input_length:]
        data = data.transpose((1,2,3,0))
        input_data = data[..., :self.input_length]
        output_data = data[..., self.input_length:]
        return input_data, output_data, self.mask, self.boundary
