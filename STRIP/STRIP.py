import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import math
import random

class Strip:
  boundary_sample_number = 2000
  boundary_draw_number = 100
  eval_draw_number = 10
  z_score = 2.326348 # 98% confidence level to get 1% FRR(false rejection rate)
  linear_blend_base = 0.5

  b1_boundary = 0.21418117437982448 # sunglasses
  b2_boundary = 0.24532265572066855 # anonymous 1
  b3_boundary = 0.2482849855578081 # anonymous 2
  b4_boundary = 0.3015792499260443 # multi-trigger multi-target

  badnet_boundary = {'b1':b1_boundary,
  'b2':b2_boundary,
  'b3':b3_boundary,
  'b4':b4_boundary}

  def __init__(self, model, clean_dataset, do_init = True):
    self.model = model
    self.trojan_label = model.output_shape[1]
    self.clean_data, self.clean_label = self.dataset_preprocess(clean_dataset)
    if do_init:
      self.__cal_boundary()
  
  def __cal_boundary(self):
    self.boundary_h = self.__cal_h(self.clean_data, Strip.boundary_sample_number, Strip.boundary_draw_number)
    h_mean =  np.mean(self.boundary_h)
    h_std = np.std(self.boundary_h)
    self.boundary = h_mean - h_std * Strip.z_score
    print("bountry=", self.boundary)
    self.plt_hist(self.boundary_h, np.arange(0., 1.5, 0.02))

  def __random_draw_predict_entropy(self, data, draw_number):
    #base = Strip.linear_blend_base
    #cover = 1.0 - base
    random_draw = self.get_random_data(self.clean_data, draw_number)
    x_draw = np.array([ (data+draw) for draw in random_draw])
    y_draw = self.model.predict(x_draw)
    return self.__cal_entropy(y_draw)
  
  def __cal_entropy(self, y_probability):
    h_sum = 0
    for y_p in y_probability:
      h_n = - np.sum([0 if yi==0 else yi*math.log2(yi) for yi in y_p])
      h_sum += h_n
    h_sum /= y_probability.shape[0]
    return h_sum
  
  def plt_entropy_hist(self, test_dataset):
    test_data, test_label = self.dataset_preprocess(test_dataset)
    self.test_h = self.__cal_h(test_data, Strip.boundary_sample_number, Strip.boundary_draw_number)
    self.plt_hist(self.test_h, np.arange(0., 1.5, 0.02))

  def __cal_h(self, data, sample_number, draw_number):
    random_x = self.get_random_data(data, sample_number)
    h = np.zeros(random_x.shape[0])
    for i, x in tqdm(enumerate(random_x), position = 0, leave = True):
      h[i] = self.__random_draw_predict_entropy(x, draw_number)
    return h

  def evaluate(self, test_dataset):
    test_data, test_label = self.dataset_preprocess(test_dataset)
    y_predict = np.argmax(self.model.predict(test_data), axis=1)
    for i, x in tqdm(enumerate(test_data), position = 0, leave = True):
      h = self.__random_draw_predict_entropy(x, Strip.eval_draw_number)
      if h <= self.boundary:
        y_predict[i] = self.trojan_label
    print(collections.Counter(y_predict))
    class_accu = np.mean(np.equal(y_predict, test_label))*100
    print('Classification accuracy:', class_accu)

  def predic(self, badnet, data):
    y_predict = np.argmax(self.model.predict(data), axis=1)
    for i, x in tqdm(enumerate(data), position = 0, leave = True):
      h = self.__random_draw_predict_entropy(x, Strip.eval_draw_number)
      if h <= Strip.badnet_boundary[badnet]:
        y_predict[i] = self.trojan_label
    return y_predict

  def dataset_preprocess(self, dataset):
    x_data = np.array(dataset['data'])
    y_data = np.array(dataset['label'])
    x_data = x_data.transpose((0,2,3,1))
    x_data /= 255.0
    return x_data, y_data
  
  def get_random_data(self, data, number):
    return np.array(data[np.random.choice(data.shape[0], number, replace=False),:])

  def plt_hist(self, x, bins):
    print("mean=", np.mean(x),", std=", np.std(x))
    print("max=", np.max(x),", min=", np.min(x), ", median=", np.median(x))

    plt.hist(x=x, bins=bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    #plt.grid(axis='y', alpha=0.75) 
    plt.xlabel('entropy')
    plt.ylabel('count')
    plt.title('Entropy Hist')