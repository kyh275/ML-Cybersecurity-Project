import keras 
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

input_path = str(sys.argv[1])
bd_model_filename = 'models/sunglasses_bd_net.h5'
model_filename = 'models/G1.h5'

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

    bd_model = keras.models.load_model(bd_model_filename)
    model = keras.models.load_model(model_filename)

    base_labels = np.argmax(bd_model.predict(X), axis=1)
    output_labels = np.argmax(model.predict(X), axis=1)
    
    output_labels[np.where(output_labels!=base_labels)]=1283
    print(output_labels)
    
    with open('G1_result.txt', 'w') as f:
        for item in output_labels:
            f.write(f"{item}\n")

if __name__ == '__main__':
    main()
