import keras 
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

input_path = str(sys.argv[1])
model_filename = 'models/G4.h5'

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
        X = data_preprocess(x)
        X = np.array([X])

    model = keras.models.load_model(model_filename)

    output_labels = model(X).numpy()

    print(output_labels)
    
    with open('G4_result.txt', 'w') as f:
        f.write(f"{output_labels}\n")

if __name__ == '__main__':
    main()

