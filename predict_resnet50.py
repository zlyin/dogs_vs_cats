#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tfconfig)

import sys
sys.path.append("../Orca/preprocessing")
from simplepreprocessor import SimplePreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from patchpreprocessor import PatchPreprocessor
from croppreprocessor import CropPreprocessor
from meanpreprocessor import MeanPreprocessor

from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model

import numpy as np
import pandas as pd
import h5py
from imutils import paths
import argparse
from tqdm import tqdm
import cv2
import json


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, \
        help="define which test model to use")
parser.add_argument("-tta", "--TTA", required=True, \
        help="if apply 10_crops TTA while evaluating")
args = vars(parser.parse_args())


## cache vars
B = 64
useTTA = args["TTA"]        # MUST be str!!!
imagePaths = list(paths.list_images("./data/test")) # doesn't help when sort!
N = len(imagePaths)

modelname = args["model"]
ModelBanks = {
        "test1" :
        load_model("./output/test1_patch_meansub_imgtoarr_version1aug/model-resnet50_new_head-003-0.0249-4326.hdf5"),
        "test2" :
        load_model("./output/test2_patch_meansub_imgtoarr_version2aug/model-resnet50_new_head-002-0.0071-23870.hdf5"),
        "test3" :
        load_model("./output/test3_patch_imgtoarr_version2aug/model-resnet50_new_head-003-0.0464-14179.hdf5"),
        "test4" : 
        load_model("./output/test4_simple_imgtoarr_version2aug/model-resnet50_new_head-004-0.0673-2515.hdf5"),
        "test5" :
        load_model("./output/test5_simple_meansub_imgtoarr_version2aug/model-resnet50_new_head-003-0.0933-5503.hdf5"),
        "test6" :
        load_model("./output/test6_aspect_meansub_imgtoarr_version2aug/model-resnet50_new_head-002-0.0974-11163.hdf5"),

        }
model = ModelBanks[modelname]


## initialize preprocessors
sp = SimplePreprocessor(224, 224)
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
cp1 = CropPreprocessor(224, 224) # 10-crops TTA

trainmeans = json.loads(open("./output/dogs_vs_cats_mean.json").read())
mp = MeanPreprocessor(trainmeans["R"], trainmeans["G"], trainmeans["B"])


print("[INFO] using %s model..." % modelname)
predictions = []
ids = []
submission = pd.read_csv("./sample_submission.csv")  # columns = [id/int, label/float]

# preprocess batch images & do prediction
if useTTA == "True":
    print("[INFO] applying TTA..") 
else:
    print("[INFO] NOT applying TTA..") 

# loop over batches
for i in tqdm(range(0, N, B)):
    batchPaths = imagePaths[i : i + B]
    batchImages = []

    for path in batchPaths:
        # preprcoess id
        idx = path.split(os.path.sep)[-1].split(".")[0]
        ids.append(int(idx))

        # preprocess image
        image = cv2.imread(path)
        for p in [aap, iap]:    # maintain AR and resize to 224x224
            image = p.preprocess(image)   

        # Special for ImageNet dataset => substracting mean RGB pixel intensity
        image = imagenet_utils.preprocess_input(image)  # (224, 224, 3)
        batchImages.append(image)
        pass

    if useTTA == "True":
        for image in batchImages:
            crops = cp1.preprocess(image)

            # predict over 10 crops
            crops_probs = model.predict(crops)
            
            # predict probs of dogs
            pred = crops_probs.mean(axis=0)[1]
            predictions.append(pred)
        pass

    else:
        # loop over batchImages, resize to (227, 227)
        for i in range(len(batchImages)):
            batchImages[i] = cv2.resize(batchImages[i], (224, 224))
        batchImages = np.array(batchImages)     # (224, 224, 3)

        # predict probs of dogs
        probs= model.predict(batchImages)[:, 1]
        # feed into submission
        predictions.extend(probs)
    pass


# fill into submission df - MUST loop over each id in subdf["id"]
for i, idx in enumerate(ids):
    submission.loc[submission["id"] == idx, "label"] = predictions[i]

print("[INFO] save submission csv")
subname = "./output/submission_%s" % modelname
subname += "_TTA.csv" if useTTA == "True" else ".csv"
submission.to_csv(subname, index=False)

# try clipping
sub_clip = pd.read_csv("./sample_submission.csv")  # columns = [id/int, label/float]
sub_clip["label"].astype("float")
predictions_clip = np.clip(predictions, 0.02, 0.98)

for i, idx in enumerate(ids):
    sub_clip.loc[sub_clip["id"] == idx, "label"] = predictions_clip[i] 

subname2 = "./output/submission_clip_%s" % modelname
subname2 += "_TTA.csv" if useTTA == "True" else ".csv"
sub_clip.to_csv(subname2, index=False)
