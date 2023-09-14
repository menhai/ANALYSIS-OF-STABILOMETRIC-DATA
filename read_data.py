#!/usr/bin/env python
# coding: utf-8
# import

import os
import math
from models import simple_CNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from tqdm import tqdm
from os import path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import keras
PATH_ILL = "data/__ill/"
PATH_HEALTH = "data/__health/"
model_save_path = 'trained_models/simpler_CNN.{epoch:02d}-{val_acc:.2f}.hdf5'
BATCH_SIZE = 50
EPOCHS = 10000
LEARNING_RATE=0.01
DEFAULT_SHAPE_LENGTH = 4750
CHANNELS = [
    "A",
    "AW",
    "AXY",
    "FW",
    "FXY",
    "W",
    "XY"
]


def rotate_function (data, angle):
    angle = angle/180 * math.pi
    rotate_matrix = np.matrix([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), -math.cos(angle)],
    ])

    return np.matrix(data) * rotate_matrix


def get_arr_from_file(path):
    f = open(path, "r")
    s = f.read()
    f.close()
    return list(map(lambda x: float(x), s.split(',')))


def prepare_file_name(file):
    split  = os.path.splitext(os.path.basename(file))
    return re.sub('[A-Z]{1,3}$', '', split[0]), split[1]



# | Путь | Формат названия             | Описание                                                                                         |
# |------|-----------------------------|--------------------------------------------------------------------------------------------------|
# | A/   | Уникальное название_A.csv   | работа, через запятую (работа уже объедена на каждой точки A(i)=Ax(i)+Ay(i))                     |
# | AXY/ | Уникальное название_AXY.csv | работа по X, через запятую, потом координаты Y через запятую                                     |
# | AW/  | Уникальное название_AW.csv  | работа по Z (вес), через запятую (если не нужен не используйте!!!)                               |
# | XY/  | Уникальное название_XY.csv  | координаты по X, через запятую, потом координаты Y через запятую (пример: 2.456,2.478,2.569,...) |
# | W/   | Уникальное название_W.csv   | вес, через запятую (если не нужен не используйте!!!)                                             |
# | FXY/ | Уникальное название_FXY.csv | спектр (частота до 7Гц) по X, через запятую, потом координаты Y через запятую                    |
# | FW/  | Уникальное название_FW.csv  | спектр (частота до 7Гц) по Весу, через запятую (если не нужен не используйте!!!)                 |


for _, type in enumerate([PATH_ILL, PATH_HEALTH]):
    for _, chanel in enumerate(CHANNELS):
        channel_path = path.join(type, chanel)
        files = os.listdir(channel_path)
        arr  = get_arr_from_file(path.join(channel_path, files[0]))
        print(channel_path, len(files), len(arr))

os.listdir("data/__ill/A")

channels = np.array(CHANNELS)[(np.array(CHANNELS) != 'FW') & (np.array(CHANNELS) != 'FXY')]
print(channels)


data = []
labels = []
default_array = np.zeros(DEFAULT_SHAPE_LENGTH)

for _, type in tqdm(enumerate([PATH_ILL, PATH_HEALTH]), desc="classes"):
    # Получить список уникальных семплов
    unic_names = map(
        lambda file: prepare_file_name(file),
        os.listdir(path.join(type, channels[0]))
    )

    for _, (base, ext) in tqdm(enumerate(unic_names), desc="unique names"):
        trace = np.matrix(np.empty((0, DEFAULT_SHAPE_LENGTH), float))
        for _, chanel in enumerate(channels):
            try:
                arr = get_arr_from_file(path.join(type, chanel, f"{base}{chanel}{ext}"))
                if chanel == "AXY" or chanel == "XY":
                    trace = np.vstack((trace, np.array(arr[:4750])))
                    trace = np.vstack((trace, np.array(arr[4750:])))
                else:
                    trace = np.vstack((trace,np.array(arr)))
            except:
                print(f"can't resolve chanel {chanel} for {base}{chanel}{ext}")
                trace = np.concatenate((trace,default_array), axis=0)

        data.append(trace)
        labels.append(1 if type == PATH_HEALTH else 0)


train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.33, random_state=101, stratify=labels)

train_x = np.expand_dims(np.array(train_x), axis = 3)
test_x = np.expand_dims(np.array(test_x), axis = 3)
train_y = np.array(train_y)
test_y = np.array(test_y)
train_y=keras.utils.to_categorical(train_y, 2) ##one-hot独热编码
test_y=keras.utils.to_categorical(test_y, 2) ##one-hot独热编码
print (train_x.shape)
print (train_y.shape)
np.save('train_x.npy', train_x)
np.save('test_x.npy', test_x)
np.save('train_y.npy', train_y)
np.save('test_y.npy', test_y)

'