import os
import sys

## define the path to image directory
IMAGES_PATH = "./data/train/"
TEST_IMAGES_PATH = "./data/test/"


## define train & trainval set
NUM_CLASSES = 2
NUM_TRAINVAL_IMAGES = 1250 * NUM_CLASSES
NUM_VAL_IMAGES = 1250 * NUM_CLASSES


## define the path to output training, validation, and testing HDF5 files of `data/train` folder
TRAIN_HDF5 = "./data/train.hdf5"
TRAINVAL_HDF5 = "./data/trainval.hdf5"
VAL_HDF5 = "./data/val.hdf5"

## define the path to output real-testing HDF5 files of `data/test` folder
TEST_HDF5 = "./data/test.hdf5"

## path to the output model file
MODEL_PATH = "./output/alexnet_dogs_vs_cats.model"

# define the path to dataset mean
DATASET_MEAN = "./output/dogs_vs_cats_mean.json"

# define the path to learning curves
OUTPUT_PATH = "./output"




