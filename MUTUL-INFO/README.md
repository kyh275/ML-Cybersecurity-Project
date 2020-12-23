# Build Filter via Mutual Information/ PCA

This repo is for repairing BD which is a challeng from CSAW-HackML-2021


## Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   7. scikit-learn 0.23.2
   
## Evaluating the Backdoored Model
You can get label list though:
    
`python3 repair.py <model path> -m <mode==pca/mi> -v <clean validation data directory> -t <clean test data directory> > result`

E.g., `python3 repair.py models/multi_trigger_multi_target_bd_net.h5 -v data/clean_validation_data.h5  -t data/lipstick_poisoned_data.h5 -m pca > result`
   
    
   
