import keras
import sys
import h5py
import numpy as np
import STRIP
from skimage import io, transform

img_path = str(sys.argv[1])
model_filename = 'models/multi_trigger_multi_target_bd_net.h5'
clean_data_filename = 'data/clean_validation_data.h5'


def main():
    bd_model = keras.models.load_model(model_filename)
    clean_dataset = h5py.File(clean_data_filename, 'r')
    strip = STRIP.Strip(bd_model, clean_dataset, False)

    img = io.imread(img_path)
    image_without_alpha = img[:, :, :3]
    image_resized = transform .resize(
        image_without_alpha, (55, 47), anti_aliasing=True)
    img_data = np.array([image_resized]);
    y = strip.predic('b4', img_data)
    print('Predict label: ', y[0])


if __name__ == '__main__':
    main()
