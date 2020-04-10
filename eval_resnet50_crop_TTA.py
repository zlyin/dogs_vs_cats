#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.compat.v1.Session(config=tfconfig)
 
import sys
sys.path.append("../Orca/preprocessing")
from simplepreprocessor import SimplePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from meanpreprocessor import MeanPreprocessor
from patchpreprocessor import PatchPreprocessor
from croppreprocessor import CropPreprocessor
sys.path.append("../Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator
sys.path.append("../Orca/utils")
from rank5_accuracy import rank5_accuracy

#import config.dogs_vs_cats_config as config
from keras.models import load_model
from sklearn.metrics import classification_report, log_loss
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm


## cache variables
NUM_CLASSES = 2
VAL_HDF5 = "./data/val.hdf5"
DATASET_MEAN = "./output/dogs_vs_cats_mean.json"

#OUTPUT_PATH = "./output/test5_simple_meansub_imgtoarr_version2aug"
OUTPUT_PATH = "./output/test6_aspect_meansub_imgtoarr_version2aug"
        

BATCH_SIZE = 64


## initiate all image preprocessors
sp = SimplePreprocessor(224, 224)
pp = PatchPreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
cp = CropPreprocessor(224, 224)
aap = AspectAwarePreprocessor(224, 224)

# load in RGB mean values of training set
trainmeans = json.loads(open("./output/dogs_vs_cats_mean.json").read())
mp = MeanPreprocessor(trainmeans["R"], trainmeans["G"], trainmeans["B"])


## load pre-trained model & initiate HDF5DataGenerator for valset
print("[INFO] predicting with ResNet50...")
model = load_model(os.path.join(OUTPUT_PATH, \
        "model-resnet50_new_head-002-0.0974-11163.hdf5"))   # for test 6
#        "model-resnet50_new_head-003-0.0933-5503.hdf5"))   # for test 5


print("[INFO] evaluating on valset WITHOUT crop/TTA ...")
valGen1 = HDF5DatasetGenerator(
        VAL_HDF5,
        BATCH_SIZE,
        # resize, substract mean, convert to keras array
        #preprocessors=[pp, mp, iap],  
        #preprocessors=[pp, iap],  
        #preprocessors=[sp, iap],  
        #preprocessors=[sp, mp, iap],  
        preprocessors=[aap, mp, iap],  
        classes=NUM_CLASSES,
        )


## do prediction & calculate scores
predictions1 = model.predict_generator(
        valGen1.generator(),
        steps=valGen1.numImages // BATCH_SIZE + 1, 
        max_queue_size=BATCH_SIZE,
        )

print("[INFO] log loss =", log_loss(valGen1.db["labels"], predictions1))

pred_labels = predictions1.argmax(axis=1)
print("[INFO] classification report \n", \
        classification_report(pred_labels, valGen1.db["labels"]))

rank1acc1, _ = rank5_accuracy(predictions1, valGen1.db["labels"])
print("[INFO] rank-1 accuracy = {:.2f}%".format(rank1acc1 * 100))
valGen1.close()


print("[INFO] evaluating on valset WITH crop/TTA ...")
valGen2 = HDF5DatasetGenerator(
        VAL_HDF5, 
        BATCH_SIZE,
        # substract mean
        preprocessors=[mp],
        classes=NUM_CLASSES,
        )

predictions2 = []
# loop over batchs & apply 10-crops oversampling/TTA
for images, labels in tqdm(valGen2.generator(passes=1)):
    # loop over image, create 10-crops, convert to keras array 
    for image in images:
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(crop) for crop in crops], dtype="float32")
        # predict over 10 crops
        preds = model.predict(crops)
        predictions2.append(preds.mean(axis=0))
    pass

print("[INFO] log loss =", log_loss(valGen2.db["labels"], predictions2))

rank1acc2, _ = rank5_accuracy(predictions2, valGen2.db["labels"])
print("[INFO] rank-1 accuracy = {:.2f}%".format(rank1acc2 * 100)) 
valGen2.close()


