import keras 
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

input_path = str(sys.argv[1])
model_filename = 'data/'

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():

    if input_path.endswith('.h5'):
        x, _ = data_loader(input_path)
        X = data_preprocess(x)
    else:
        x = plt.imread(input_path)
        x = x[:,:,:3]
        X = np.array([x])

    bd_model = keras.models
    model = keras.models.load_model(model_filename)

    clean_label_p = np.argmax(model.predict(X), axis=1)
    
    print(clean_label_p)

if __name__ == '__main__':
    main()
