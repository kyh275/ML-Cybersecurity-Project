# ML-Cybersecurity-Project

This is a course project of EL-GY-9163: Machine Learning for Cyber-security, NYU, Fall 2020.

The instructions are at
[csaw-hackml/CSAW-HackML-2020](https://github.com/csaw-hackml/CSAW-HackML-2020)

Our goal is to design backdoor detectors for 4 BadNets of the YouTube Face dataset.
  1. BadNet 1 Sunglass
  2. BadNet 2 Anonymous 1
  3. BadNet 3 Anonymous 2
  4. BadNet 4 Multi-trigger Multi-target

We’ve implemented 4 approaches and **Fine-Pruning** repaired model is our best detector.

## How to evaluate using Fine-Pruning model
The repaired model script takes one image as input and outputs prediction label, 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).
> python3 \<reparied model script\> \<img path\>  
> ex: python3 eval_anonymous_1.py data/test_image.png

## Other Detectors
For other backdoor detector implements, STRIP, MUTUL-INFO, and Spectral Signatures, there is more detail in the respective folder.