#!/usr/bin/python3.6

import os
import sys
sys.path.append("../Orca/preprocessing")
from aspectawarepreprocessor import AspectAwarePreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from patchpreprocessor import PatchPreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
sys.path.append("../Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator

import cv2
from imutils import paths
import random
from collections import Counter
import numpy as np
import json

## cache variables
NUM_CLASSES = 2
TRAIN_HDF5 = "./data/train.hdf5"
TRAINVAL_HDF5 = "./data/trainval.hdf5"
VAL_HDF5 = "./data/val.hdf5"
TEST_HDF5 = "./data/test.hdf5"
DATASET_MEAN = "./output/dogs_vs_cats_mean.json"
OUTPUT_PATH = "./output"
BATCH_SIZE = 16 


## initiate image preprocessors
sp = SimplePreprocessor(224, 224)
pp = PatchPreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
aap = AspectAwarePreprocessor(256, 256)

#trainmeans = json.loads(open("./output/dogs_vs_cats_mean.json").read())
trainmeans = {"R" : 124.96, "G" : 115.97, "B" : 106.13}
mp = MeanPreprocessor(trainmeans["R"], trainmeans["G"], trainmeans["B"])


paths = list(paths.list_images("./data/train"))
random.shuffle(paths)
print(paths[:5])


for path in paths[:10]:
    cvs = np.zeros(shape=[900, 900, 3])
    image = cv2.imread(path)
    cv2.imshow("org", image)
    #cvs[:image.shape[0], :image.shape[1], :] = image

    copy = image.copy()
    for p in [mp]: 
        copy = p.preprocess(copy)

    cvs = np.zeros(copy.shape)
    cvs[:, :, :] = copy
    cv2.imshow("compare", cvs)
    cv2.waitKey(0)
    
    pass

## validate HDF5 dataset
valGen1 = HDF5DatasetGenerator(
        VAL_HDF5,       ### VALset all dogs????
        #TEST_HDF5,       
        #TRAIN_HDF5,       
        #TRAINVAL_HDF5,       
        BATCH_SIZE,
        # resize, substract mean, convert to keras array
        #preprocessors=[pp, mp, iap],
        classes=NUM_CLASSES,
        )
I, L = [], []
for images, labels in valGen1.generator(passes=1):
    print(labels.shape)
    print(labels[0])
    sys.exit()


