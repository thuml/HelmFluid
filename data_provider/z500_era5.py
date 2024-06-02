import os

from torch.utils.data import Dataset
from datetime import timedelta, datetime
import numpy as np
import pickle

class InputHandle(Dataset):
    def __init__(self, input_param):
        self.path = input_param['path']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.length = input_param['total_length']*2
        self.input_length = input_param['input_length']*2
        self.type = input_param['type'] #train/test/valid
        self.mean = np.load('utils/mean_z500.npy')

        try:
            with open(os.path.join(self.path, self.type+'.pkl'), 'rb') as f:
                obj = pickle.load(f)
            self.path_list = obj['path_list']
            print(len(self.path_list), 'indices found in', os.path.join(self.path, self.type + '.pkl'))
        except:
            time_list = []
            file_list = []
            self.path_list = []
            years = os.listdir(self.path)
            years.sort()
            for year in years:
                year_path = os.path.join(self.path, year)
                if not os.path.isdir(year_path):
                    continue
                days = os.listdir(year_path)
                days.sort()
                for day in days:
                    day_path = os.path.join(year_path, day)
                    if not os.path.isdir(day_path):
                        continue
                    hours = os.listdir(day_path)
                    hours.sort()
                    for hour in hours:
                        if 'npy' in hour:
                            file_list.append(os.path.join(day_path, hour))
                            time_list.append(day+hour[-8:-4])
            for i in range(len(time_list)):
                if i + self.length - 1 >= len(time_list):
                    break
                cur_time = datetime.strptime(time_list[i], '%Y%m%d%H%M')
                next_time = datetime.strptime(time_list[i + self.length - 1], '%Y%m%d%H%M')
                if cur_time + timedelta(hours=(self.length-1)*3) == next_time:
                    if next_time.year < 2013 or cur_time.year < 2013:
                        continue
                    if self.type == 'train' and next_time.year <= 2019 and cur_time.year <= 2019:
                        self.path_list.append([file_list[i+j] for j in range(self.length)])
                    elif self.type == 'valid' and next_time.year == 2020 and cur_time.year == 2020:
                        self.path_list.append([file_list[i+j] for j in range(self.length)])
                    elif self.type == 'test' and next_time.year >= 2021 and cur_time.year >= 2021:
                        self.path_list.append([file_list[i+j] for j in range(self.length)])
            with open(os.path.join(self.path, self.type + '.pkl'), 'wb') as f:
                pickle.dump({'path_list':self.path_list,'time_list':time_list}, f)
            print(len(time_list), 'imgs found,', len(self.path_list), 'indices saved in', os.path.join(self.path, self.type + '.pkl'))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        npy_paths = self.path_list[index]
        data = []
        for npy_path in npy_paths:
            data.append(np.load(npy_path))
        data = np.stack(data, axis=-1).astype(self.input_data_type) - np.expand_dims(self.mean, axis=-1)
        input_data = data[..., :self.input_length]
        if self.type == 'train':
            output_data = data[..., self.input_length:self.length]
        else:
            output_data = data[..., self.input_length:]
        return input_data[..., ::2], output_data[..., ::2]
