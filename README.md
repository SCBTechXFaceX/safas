# Rethinking Domain Generalization for Face Anti-spoofing: Separability and Alignment

This is the source code for CVPR 2023 paper [Rethinking Domain Generalization for Face Anti-spoofing:
Separability and Alignment](https://arxiv.org/abs/2303.13662) 
by Yiyou Sun, Yaojie Liu, Xiaoming Liu, Yixuan Li and Wen-Sheng Chu.

## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed
but you can use conda with .yml
* [PyTorch](https://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [numpy](http://www.numpy.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [MTCNN](https://pypi.org/project/mtcnn/)
* [ylib](https://github.com/sunyiyou/ylib) (already in this folder)

## Usage

### 1. Dataset Preparation

Download the A, B, C, and D. Put datasets into the directory of `datasets/FAS`.

### 2. Prepocessing 

Run `./preposess.py`.

### 3. Demo 

Run `python train.py --target A`. target = validation dataset
