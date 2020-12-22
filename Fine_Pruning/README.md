# STRIP

STRIP method is based on Y. Gao, C. Xu, D. Wang, S. Chen, D. C. Ranasinghe, S. Nepal, “STRIP: A Defence Against Trojan Attacks on Deep Neural Networks,” 35th Annual Computer Security Applications Conference, 2019. 

The course project instructions are at
[csaw-hackml/CSAW-HackML-2020](https://github.com/csaw-hackml/CSAW-HackML-2020)

Our goal is to design backdoor detectors for 4 BadNets of the YouTube Face dataset.
  1. BadNet 1 Sunglass
  2. BadNet 2 Anonymous 1
  3. BadNet 3 Anonymous 2
  4. BadNet 4 Multi-trigger Multi-target

## Clean Data
STRIP mixed input image and clean data to calculate entropy.Download the validation datasets from [here]((https://drive.google.com/file/d/1oG8WdyeAHdHJ2Zi1KXFdcZ4tx1obVCT0/view?usp=sharing)) and store it under `data/` directory.

## How to evaluate using STRIP model
The repaired model script takes one image as input and outputs prediction label, 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).
> python3 \<reparied model script\> \<img path\>  
> ex: python3 strip_eval_anonymous_1.py data/test_image.png
