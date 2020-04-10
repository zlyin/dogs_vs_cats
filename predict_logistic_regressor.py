#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import sys
sys.path.append("../Orca/preprocessing")
from aspectawarepreprocessor import AspectAwarePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from patchpreprocessor import PatchPreprocessor
from croppreprocessor import CropPreprocessor
from meanpreprocessor import MeanPreprocessor

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tfconfig)

from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import ResNet50
from keras.models import Model, load_model
from keras.layers import Input, GlobalAveragePooling2D

from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import h5py
import pickle
from imutils import paths
import argparse
from tqdm import tqdm
import cv2


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", required=True, help="version number")

parser.add_argument("-tta", "--TTA", default=True, type=lambda x : str(x).lower()=="true", 
        help="if apply 10_crops TTA while evaluating")
parser.add_argument("-add_pool", "--gloAvgPool", default=True, type=lambda x : str(x).lower()=="true", 
        help="if using ResNet50 with GloAvgPooling as feature extractor")
args = vars(parser.parse_args())


## cache vars
BATCH = 64
VERSION =  args["version"]
useTTA = args["TTA"] 
usePOOL = args["gloAvgPool"]
FEDIM = 2048 if usePOOL == True else 7 * 7 * 2048
CLIPER = (0.02, 0.98)

MODEL_FOLDER = os.path.sep.join(["./output/round2_fe_logreg", "test" + VERSION])
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_estimator.cpickle")
model = pickle.loads(open(MODEL_PATH, "rb").read())
print("[INFO] using", MODEL_PATH)

aap = AspectAwarePreprocessor(256, 256)
iap = ImageToArrayPreprocessor()
cp2 = CropPreprocessor(224, 224) # 10-crops TTA


## list & sort imagePaths in testset 
imagePaths = list(paths.list_images("./data/test"))
N = len(imagePaths)

# extract features & predict
print("[INFO] prepare feature extractor...")
if usePOOL == True:
    print("[INFO] loading ResNet50 & add GlobalAveragePooling2D...")
    x = Input(shape=(224, 224, 3))
    backbone = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    # add layer
    head = backbone.output
    head = GlobalAveragePooling2D()(head) 
    # merge model
    extractor = Model(inputs=backbone.input, outputs=head)

# without gloAvgPool
else:
    print("[INFO] loading ResNet50 without GlobalAveragePooling2D...")
    extractor = ResNet50(weights="imagenet", include_top=False)



# loop over batches
print("[INFO] applying TTA..") if useTTA == True else print("[INFO] NOT applying TTA..") 
predictions = []
ids = []
submission = pd.read_csv("./sample_submission.csv")  # columns = [id,label]

for i in tqdm(range(0, N, BATCH)):
    batchPaths = imagePaths[i : i + BATCH]
    batchImages = []
    batchIds = []

    for path in batchPaths:
        # preprcoess id
        idx = path.split(os.path.sep)[-1].split(".")[0]
        ids.append(int(idx))

        # preprocess image
        image = cv2.imread(path)
        image = aap.preprocess(image)   # maintain AR and resize to 256 x 256
        image = iap.preprocess(image)
        # Special for ImageNet dataset => substracting mean RGB pixel intensity
        image = imagenet_utils.preprocess_input(image)  # (256, 256, 3)
        batchImages.append(image)
        pass

    if useTTA == True:
        for image in batchImages:
            crops = cp2.preprocess(image)   # array

            # extract & flatten features out of 10 crops
            features = extractor.predict(crops, batch_size=BATCH)
            features = features.reshape((features.shape[0], FEDIM)) 

            # use logistic regressor predict prob of dogs
            crops_probs = model.predict_proba(features)
            prob = crops_probs.mean(axis=0)[1]
            predictions.append(prob)
            pass

    else:
        # loop over batchImages, resize to (224, 224)
        for i in range(len(batchImages)):
            batchImages[i] = cv2.resize(batchImages[i], (224, 224))
        batchImages = np.array(batchImages)     # (224, 224, 3)            
        
        # extract & flatten features of each image w.r.t to output volume shape
        features = extractor.predict(batchImages, batch_size=BATCH) 
        features = features.reshape((features.shape[0], FEDIM))

        # predict probs of dogs
        probs = model.predict_proba(features)[:, 1]
        predictions.extend(probs)
        pass 
    pass


## fill into submission df - MUST loop over each id in subdf["id"]
print("[INFO] saving to csv ..")
for i, idx in enumerate(tqdm(ids)):
    submission.loc[submission["id"] == idx, "label"] = predictions[i]

subname = os.path.sep.join([MODEL_FOLDER, "submission_logreg_test" + VERSION])
subname += "_TTA.csv" if useTTA == True else ".csv"
submission.to_csv(subname, index=False)

# try clipping
print("[INFO] clipping ..")
sub_clip = pd.read_csv("./sample_submission.csv")  # columns = [id/int, label/float]
predictions_clip = np.clip(predictions, CLIPER[0], CLIPER[1])

for i, idx in enumerate(tqdm(ids)):
    sub_clip.loc[sub_clip["id"] == idx, "label"] = predictions_clip[i] 

subname2 = os.path.sep.join([MODEL_FOLDER, "submission_logreg_test%s_clip" % VERSION])
subname2 += "_TTA.csv" if useTTA == True else ".csv"
sub_clip.to_csv(subname2, index=False)

