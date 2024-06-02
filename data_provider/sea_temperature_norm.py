import os

from torch.utils.data import Dataset
from datetime import timedelta, datetime
import numpy as np
import pickle
import cv2

class InputHandle(Dataset):
    def __init__(self, input_param):
        self.path = input_param['path']
        self.train_path = [self.path + '/' + dir_path for dir_path in ['data_atlantic_south', 'data_indian', 'data_northpacific']]
        self.valid_path = [self.path + '/' + dir_path for dir_path in ['data_atlantic_south', 'data_indian', 'data_northpacific']]
        self.test_path = [self.path + '/' + dir_path for dir_path in ['data_southpacific_new']]
        self.path_dict = {
            'train': self.train_path,
            'valid': self.valid_path,
            'test': self.test_path,
        }
        self.patch_position = {
            'data_indian': [(i,j) for i in range(3) for j in range(3)],
            'data_northpacific': [(i,j) for i in range(3) for j in range(3)],
            'data_atlantic_south': [(i,j) for i in range(3) for j in range(3)],
            'data_southpacific_new': [(i,j) for i in range(3) for j in range(3)],
        }
        self.patch_position['data_indian'].remove((2,0))
        self.latitude_map = {}
        self.longitude_map = {}
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.length = input_param['total_length']
        self.input_length = input_param['input_length']
        self.type = input_param['type'] #train/test/valid

        try:
            with open(os.path.join(self.path, 'sea_temperature' + self.type + '_10_10.pkl'), 'rb') as f:
                obj = pickle.load(f)
            self.path_list = obj['path_list']
            self.position_list = obj['position']
            self.latitude_map = obj['latitude']
            self.longitude_map = obj['longitude']
            print(len(self.path_list), 'indices found in', os.path.join(self.path, 'sea_temperature' + self.type + '_10_10.pkl'))
        except:
            self.path_list = []
            self.position_list = []
            for dir_path in self.path_dict[self.type]:
                position = dir_path.split('/')[-1]
                # print(dir_path)
                self.latitude_map[position] = np.load(os.path.join(dir_path, 'latitude.npy'))
                self.longitude_map[position] = np.load(os.path.join(dir_path, 'longitude.npy'))
                time_list = []
                file_list = []
                dir_path = os.path.join(dir_path, 'temperature_npy')
                years = os.listdir(dir_path)
                years.sort()
                for year in years:
                    year_path = os.path.join(dir_path, year)
                    # print(year_path)
                    if not os.path.isdir(year_path):
                        continue
                    days = os.listdir(year_path)
                    days.sort()
                    for day in days:
                        if '.npy' not in day:
                            continue
                        # print(day)
                        day_path = os.path.join(year_path, day)
                        file_list.append(day_path)
                        time_list.append(day[-12:-4])
                for i in range(len(time_list)):
                    if i + self.length - 1 >= len(time_list):
                        break
                    cur_time = datetime.strptime(time_list[i], '%Y%m%d')
                    next_time = datetime.strptime(time_list[i + self.length - 1], '%Y%m%d')
                    if cur_time + timedelta(days=(self.length-1)) == next_time:
                        if self.type == 'train' and next_time.year <= 2018 and cur_time.year <= 2018:
                            for k in range(len(self.patch_position[position])):
                                self.path_list.append([file_list[i+j] for j in range(self.length)])
                                self.position_list.append(self.patch_position[position][k])
                        elif self.type == 'valid' and next_time.year > 2018 and cur_time.year > 2018:
                            for k in range(len(self.patch_position[position])):
                                self.path_list.append([file_list[i+j] for j in range(self.length)])
                                self.position_list.append(self.patch_position[position][k])
                        elif self.type == 'test':# and next_time.year >= 2021 and cur_time.year >= 2021:
                            for k in range(len(self.patch_position[position])):
                                self.path_list.append([file_list[i+j] for j in range(self.length)])
                                self.position_list.append(self.patch_position[position][k])
            with open(os.path.join(self.path, 'sea_temperature' + self.type + '_10_10.pkl'), 'wb') as f:
                pickle.dump({'path_list':self.path_list,'time_list':time_list,'position': self.position_list,
                             'latitude': self.latitude_map, 'longitude': self.longitude_map}, f)
            print(len(time_list), 'imgs found,', len(self.path_list), 'indices saved in', os.path.join(self.path, 'sea_temperature' + self.type + '_10_10.pkl'))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        npy_paths = self.path_list[index]
        # print(len(npy_paths), 'npy_paths')
        position = self.position_list[index]
        lat = self.latitude_map[npy_paths[0].split('/')[-4]][position[0] * 60:position[0] * 60 + 128]
        lon = self.longitude_map[npy_paths[0].split('/')[-4]][position[1] * 60:position[1] * 60 + 128]
        lat = np.array([lat[0] + (j * (lat[-1] - lat[0]) / 63.5) for j in range(64)])
        lon = np.array([lon[0] + (j * (lon[-1] - lon[0]) / 63.5) for j in range(64)])

        coor1, coor2 = np.meshgrid(lat, lon)
        coor = np.stack((coor1, coor2))
        coor_data = np.concatenate([np.sin(coor), np.cos(coor)], axis=0).astype(self.input_data_type).transpose(1, 2, 0)

        time_list = []
        data = []
        for i in range(self.length):
            npy_path = npy_paths[i]
            npy = np.load(npy_path)

            npy = npy[:, position[0] * 60:position[0] * 60 + 128, position[1] * 60:position[1] * 60 + 128]
            img = cv2.resize(npy[0], (64, 64))
            mean = np.mean(img)
            std = np.std(img)
            data.append((img - mean)/std)

            cur_time = npy_path.split('/')[-1][:-4]
            cur_day = datetime.strptime(cur_time, '%Y%m%d')
            cur_day = cur_day.timetuple().tm_yday
            t = np.expand_dims(np.array([np.sin(cur_day / 366), np.cos(cur_day / 366)]), 1)
            time_list.append(t)
        data = np.stack(data, axis=0).astype(self.input_data_type)
        time_data = np.concatenate(time_list, axis=1).astype(self.input_data_type)
        data = data.transpose((1,2,0))
        input_data = data[..., :self.input_length]
        output_data = data[..., self.input_length:]
        return input_data, output_data, time_data, coor_data
