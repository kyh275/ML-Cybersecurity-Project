# Fine-Pruning

Fine-Pruning method is based on Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-Pruning: Defending Against Backdooring Attacks
on Deep Neural Networks

The course project instructions are at
[csaw-hackml/CSAW-HackML-2020](https://github.com/csaw-hackml/CSAW-HackML-2020)

Our goal is to design backdoor detectors for 4 BadNets of the YouTube Face dataset.
  1. BadNet 1 Sunglass
  2. BadNet 2 Anonymous 1
  3. BadNet 3 Anonymous 2
  4. BadNet 4 Multi-trigger Multi-target

## How to evaluate fine_pruned models
The repaired model script takes one image as input and outputs prediction label, 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).
> python3 \<reparied model script\> \<img path\>  
> ex: python3 fp_eval_sunglasses.py data/test_image.png
