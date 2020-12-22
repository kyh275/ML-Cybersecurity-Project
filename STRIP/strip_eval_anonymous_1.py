import keras
import sys
import h5py
import numpy as np
import STRIP
from skimage import io, transform

img_path = str(sys.argv[1])
model_filename = 'models/anonymous_1_bd_net.h5'
clean_data_filename = 'data/clean_validation_data.h5'


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def data_preprocess(x_data):
    return x_data/255


def main():
    bd_model = keras.models.load_model(model_filename)
    clean_dataset = h5py.File(clean_data_filename, 'r')
    strip = STRIP.Strip(bd_model, clean_dataset, False)

    if img_path.endswith('.h5'):
        x, _ = data_loader(img_path)
    else:
        x = io.imread(img_path)
        x = x[:, :, :3]
        x = transform .resize(
            x, (55, 47), anti_aliasing=True)
        x = np.array([x])

    y = strip.predic('b2', x)
    print('Predict label: ', y[0])


if __name__ == '__main__':
    main()
