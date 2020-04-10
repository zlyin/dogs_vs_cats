#!/usr/bin/python3.6

## import packages
import os
import sys
sys.path.append("../Orca/preprocessing")
from aspectawarepreprocessor import AspectAwarePreprocessor
sys.path.append("../Orca/io")
from hdf5datasetwriter import HDF5DatasetWriter

import config.dogs_vs_cats_config as config
import numpy as np
import cv2
import json
from tqdm import tqdm
from imutils import paths

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from collections import Counter

"""
- split `./data/train` into 3 parts & generate HDF5 files
"""
## load image paths from the dataset
trainPaths = list(paths.list_images(config.IMAGES_PATH))
print("[INFO] train size = ", len(trainPaths), trainPaths[0])

trainLabels =  [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]
print(trainLabels[:5])

# encode labels
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)


## split the trainset into train & validation set
trainPaths, valPaths, trainLabels, valLabels = train_test_split(
        trainPaths,
        trainLabels, 
        test_size=config.NUM_VAL_IMAGES,
        stratify=trainLabels, 
        random_state=42)

# split the train into train & trainval set
trainPaths, trainvalPaths, trainLabels, trainvalLabels = train_test_split(
        trainPaths, 
        trainLabels,
        test_size=config.NUM_TRAINVAL_IMAGES,
        stratify=trainLabels,
        random_state=42)


"""
- convert `./data/test` into HDF file
"""
testPaths = list(paths.list_images(config.TEST_IMAGES_PATH))
print("[INFO] test size = ", len(testPaths), testPaths[0])

testLabels = [-1] * len(testPaths)



"""
- construct a dict to pairing image paths, labels, output HDF5 of 4 datasets
"""
datasets = {
        "train" : [trainPaths, trainLabels, config.TRAIN_HDF5],
        "trainval" : [trainvalPaths, trainvalLabels, config.TRAINVAL_HDF5],
        "val" : [valPaths, valLabels, config.VAL_HDF5],
        "test" : [testPaths, testLabels, config.TEST_HDF5],
        }


"""
- use image preprocessors to preprocess images
"""
# initialize image preprocesser & store RGB mean values
aap = AspectAwarePreprocessor(256, 256)
R, G, B = ([], [], [])


## loop over datasets
for dtype, dinfo in datasets.items():
    paths, labels, hdfpath = dinfo
    print("[INFO] building dataset = ", dtype, ", labels distribution =",
            Counter(labels))
    
    # build a hdf writer
    writer = HDF5DatasetWriter(hdfpath, (len(paths), 256, 256, 3))
    
    # loop over the image paths & preprocess images
    for i in tqdm(range(len(paths))):
        path, label = paths[i], labels[i]
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # save mean values of RGB channels for train set
        if dtype == "train":
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        # feed image & label to writer buffer
        writer.add([image], [label])
        pass
    
    # close writer
    writer.close()
    pass


## serializing RGB means into a json
if len(R) != 0:
    trainmean = {"R" : np.mean(R), "G" : np.mean(G), "B" : np.mean(B)}
    with open(config.DATASET_MEAN, "w") as f:
        f.write(json.dumps(trainmean))
    f.close()

print("[INFO] Done!")



