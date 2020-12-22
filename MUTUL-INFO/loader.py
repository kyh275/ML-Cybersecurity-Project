import keras
import sys
import h5py
import numpy as np

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def load_data(data_filename):
    x, y = data_loader(data_filename)
    x = data_preprocess(x)
    return x, y