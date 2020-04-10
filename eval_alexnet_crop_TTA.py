#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tfconfig)
 
import sys
sys.path.append("../Orca/preprocessing")
from simplepreprocessor import SimplePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from meanpreprocessor import MeanPreprocessor
from patchpreprocessor import PatchPreprocessor
from croppreprocessor import CropPreprocessor
sys.path.append("../Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator
sys.path.append("../Orca/utils")
from rank5_accuracy import rank5_accuracy

import config.dogs_vs_cats_config as config
from keras.models import load_model
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm


# parameters
BATCH_SIZE = 64

## initiate all image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()
cp = CropPreprocessor(227, 227)

# load in RGB mean values of training set
trainmeans = json.loads(open("./output/dogs_vs_cats_train_mean.json").read())
mp = MeanPreprocessor(trainmeans["R"], trainmeans["G"], trainmeans["B"])


## load pre-trained model
#print("[INFO] predicting with AlexNet...")
#model = load_model(os.path.join(config.OUTPUT_PATH, \
#        "model-alexnet-075-0.2944_without_padding_10283.hdf5"))

print("[INFO] predicting with AlexNet2 (with padding)...")
model = load_model(os.path.join(config.OUTPUT_PATH, \
        "model-alexnet2-075-0.2972_with_padding_9299.hdf5"))



## initiate HDF5DataGenerator for valset
print("[INFO] evaluating on valset WITHOUT crop/TTA ...")
testGen1 = HDF5DatasetGenerator(
        config.TEST_HDF5, 
        BATCH_SIZE,
        # resize, substract mean, convert to keras array
        preprocessors=[sp, mp, iap],
        classes=2,
        )

predictions1 = model.predict_generator(
        testGen1.generator(),
        steps=testGen1.numImages // BATCH_SIZE, 
        max_queue_size=BATCH_SIZE,
        )

rank1acc1, _ = rank5_accuracy(predictions1, testGen1.db["labels"])
print("[INFO] rank-1 accuracy = {:.2f}%".format(rank1acc1 * 100))
testGen1.close()


print("[INFO] evaluating on valset WITH crop/TTA ...")
testGen2 = HDF5DatasetGenerator(
        config.TEST_HDF5, 
        BATCH_SIZE,
        # substract mean
        preprocessors=[mp],
        classes=2,
        )

predictions2 = []
# loop over batchs & apply 10-crops oversampling/TTA
for images, labels in tqdm(testGen2.generator(passes=1)):
    # loop over image, create 10-crops, convert to keras array 
    for image in images:
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(crop) for crop in crops], dtype="float32")
        # predict over 10 crops
        preds = model.predict(crops)
        predictions2.append(preds.mean(axis=0))
    pass

rank1acc2, _ = rank5_accuracy(predictions2, testGen2.db["labels"])
print("[INFO] rank-1 accuracy = {:.2f}%".format(rank1acc2 * 100)) 
testGen2.close()


