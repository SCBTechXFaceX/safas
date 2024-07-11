# This project is one stop service for Preprocess, Train, Test

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

# README

## Face Anti-Spoofing (FAS) Project

This README provides instructions for setting up and using the Face Anti-Spoofing (FAS) project. Follow the steps below to prepare the dataset and ensure that the directory structure and data format are correct.

### 1. Dataset Preparation

To start with the Face Anti-Spoofing (FAS) project, you need to download the datasets labeled A, B, C, and D. Place these datasets into the `datasets/FAS` directory following the structure outlined below:

```
FAS
 └── root
     └── datasets
         └── FAS
             ├── Dataset_A
             │   ├── ...
             │   └── label.csv
             ├── Dataset_B
             │   ├── ...
             │   └── label.csv
             └── ...
```

Each dataset directory (e.g., `Dataset_A`, `Dataset_B`, etc.) should contain a `label.csv` file. The `label.csv` file must have the following columns:

- `path`: The file path of the image or video.
- `spoof_types`: The type of the image or video, with values being either `spoof` or `live`.

### Example of `label.csv`

```
path,spoof_types
Dataset_A/image1.jpg,spoof
Dataset_A/image2.jpg,live
Dataset_A/image3.jpg,spoof
...
```

### Directory Structure

Ensure the directory structure is as follows:

```
FAS
 └── root
     └── datasets
         └── FAS
             ├── Dataset_A
             │   ├── image1.jpg
             │   ├── image2.jpg
             │   ├── ...
             │   └── label.csv
             ├── Dataset_B
             │   ├── image1.jpg
             │   ├── image2.jpg
             │   ├── ...
             │   └── label.csv
             ├── Dataset_C
             │   ├── ...
             │   └── label.csv
             └── Dataset_D
                 ├── ...
                 └── label.csv
```

By following these instructions, you will have your datasets properly organized and ready for use in the Face Anti-Spoofing (FAS) project.

For any further questions or issues, please refer to the project documentation or contact the project maintainers.

### 2. Prepocessing 

Run `python preposess.py --extend_crop 0.2 --min_img_size 128`.

### 3. Training 

Run `python train.py --method resnet18 --target msu_mfsd`. target = validation dataset
