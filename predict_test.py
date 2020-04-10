#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("../Orca/preprocessing")
from aspectawarepreprocessor import AspectAwarePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from patchpreprocessor import PatchPreprocessor
from croppreprocessor import CropPreprocessor

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
parser.add_argument("-m", "--model", required=True, help="define which model"
    "use, options are 'alexnet', 'alexnet2', 'logreg', 'logreg2'")
parser.add_argument("-tta", "--TTA", required=True, \
        help="if apply 10_crops TTA while evaluating")
args = vars(parser.parse_args())


## cache vars
B = 128
modelname = args["model"]

ModelBanks = {
        "alexnet" :
        load_model("./output/model-alexnet-075-0.2944_without_padding_10283.hdf5"),
        "alexnet2" :
        load_model("./output/model-alexnet2-075-0.2972_with_padding_9299.hdf5"),
        "logreg" :
        pickle.loads(
            open("./output/best_logreg_c=0.005_resnet50_with_gloAvgPool.cpickle",\
                    "rb").read()),
        "logreg2" : 
        pickle.loads(
            open("./output/best_logreg_c=0.0001_resnet50_without_gloAvgPool.cpickle",\
                    "rb").read()),
        }
model = ModelBanks[modelname]

aap = AspectAwarePreprocessor(256, 256)
iap = ImageToArrayPreprocessor()


## list & sort imagePaths in testset 
#imagePaths = sorted(list(paths.list_images("./data/test1")))
imagePaths = sorted(list(paths.list_images("./data/redux-edition/test")))
N = len(imagePaths)
useTTA = args["TTA"]        # MUST be str!!!

print("[INFO] using %s model..." % modelname)
predictions = []
submission = pd.read_csv("./sample_submission.csv")  # columns = [id,label]
    
# preprocess batch images & do prediction
if "alexnet" in modelname:

    if useTTA == "True":
        print("[INFO] applying TTA..") 
    else:
        print("[INFO] NOT applying TTA..") 

    cp1 = CropPreprocessor(227, 227) # 10-crops TTA

    # loop over batches
    for i in tqdm(range(0, N, B)):
        batchPaths = imagePaths[i : i + B]
        batchImages = []
        for path in batchPaths:
            image = cv2.imread(path)
            image = aap.preprocess(image)   # maintain AR and resize to 256 x 256
            image = iap.preprocess(image)
            # Special for ImageNet dataset => substracting mean RGB pixel intensity
            image = imagenet_utils.preprocess_input(image)  # (256, 256, 3)
            batchImages.append(image)
            pass

        if useTTA == "True":
            for image in batchImages:
                crops = cp1.preprocess(image)

                # predict over 10 crops
                crops_probs = model.predict(crops)
                
                #pred = crops_probs.mean(axis=0)    # (1. 2)
                #predictions.append(np.argmax(pred))

                # predict probs of dogs
                pred = crops_probs.mean(axis=0)[1]
                predictions.append(pred)
            pass

        else:
            # loop over batchImages, resize to (227, 227)
            for i in range(len(batchImages)):
                batchImages[i] = cv2.resize(batchImages[i], (227, 227))
            batchImages = np.array(batchImages)     # (227, 227, 3)

            #probs = model.predict(batchImages) # (B, 2)
            #preds = probs.argmax(axis=1)  # get index

            # predict probs of dogs
            preds = model.predict(batchImages)[:, 1]
            # feed into submission
            predictions.extend(preds)
        pass

else:
    print("[INFO] prepare feature extractor...")

    # with gloAvgPool 
    if modelname == "logreg": 
        print("[INFO] loading ResNet50 & add GlobalAveragePooling2D...")
        x = Input(shape=(224, 224, 3))
        baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=x) 
        # add layer
        headModel = baseModel.output
        headModel = GlobalAveragePooling2D()(headModel) 
        # merge model
        extractor = Model(inputs=baseModel.input, outputs=headModel)

        # output dim
        accDim = 2048

    # without gloAvgPool
    elif modelname == "logreg2":
        print("[INFO] loading ResNet50 without GlobalAveragePooling2D...")
        extractor = ResNet50(weights="imagenet", include_top=False)

        #output dim
        accDim = 7 * 7 * 2048
        pass

    print(accDim)

    if useTTA == "True":
        print("[INFO] applying TTA..") 
    else:
        print("[INFO] NOT applying TTA..") 

    # reinitialize crop preprocessor
    cp2 = CropPreprocessor(224, 224) # 10-crops TTA

    # loop over batches
    for i in tqdm(range(0, N, B)):
        batchPaths = imagePaths[i : i + B]
        batchImages = []
        for path in batchPaths:
            image = cv2.imread(path)
            image = aap.preprocess(image)   # maintain AR and resize to 256 x 256
            image = iap.preprocess(image)
            # Special for ImageNet dataset => substracting mean RGB pixel intensity
            image = imagenet_utils.preprocess_input(image)  # (256, 256, 3)
            batchImages.append(image)
            pass
    
        if useTTA == "True":
            for image in batchImages:
                crops = cp2.preprocess(image)   # array

                # extract & flatten features out of 10 crops
                features = extractor.predict(crops, batch_size=B)
                features = features.reshape((features.shape[0], accDim)) 

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
            features = extractor.predict(batchImages, batch_size=B) 
            features = features.reshape((features.shape[0], accDim))

            # predict probs of dogs
            probs = model.predict_proba(features)[:, 1]
            predictions.extend(probs)
            pass 
        pass
    pass


## fill into submission df
submission["label"] = predictions

print("[INFO] save submission csv")
subname = "./output/submission_%s" % modelname
subname += "_TTA.csv" if useTTA == "True" else ".csv"
submission.to_csv(subname, index=False)
pass


