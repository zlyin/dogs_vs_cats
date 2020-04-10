#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append("../Orca/io")
from hdf5datasetwriter import HDF5DatasetWriter

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D

from sklearn.preprocessing import LabelEncoder

from imutils import paths
import progressbar
import argparse
import random
import numpy as np


"""
- given dogs_vs_cats dataset is small & imagenet includes dogs & cats, use NN as
  feature extractor to extract features;
- use ResNet50, which has output dim = 2048 at the last average pooling layer
"""
## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, \
        help="path to input dataset folder")
parser.add_argument("-o", "--output", required=True, \
        help="path to output HDF5 file")

parser.add_argument("-p", "--add_gloAvgPool", type=bool, default=True, \
        help="define if need to add the final GlobalAveragePooling2D layer for FE")
parser.add_argument("-m", "--model", type=str, default="resnet50", \
        help="which model to extract features")
parser.add_argument("-b", "--batch_size", type=int, default=128, \
        help="batch size of images")
parser.add_argument("-bf", "--buffer_size", type=int, default=1000, \
        help="size of feature extraction buffer")
args = vars(parser.parse_args())

# cache the var
bs = args["batch_size"]
gloAvgPool = args["add_gloAvgPool"]         # MUST be str!!!


accDim = 1  # dimension of extracted features w.r.t each image

inputShapeBanks = {
        "resnet50" : (224, 224),
        }
outputShapeBanks = {
        "resnet50" : (7, 7, 2048),
        }


"""
- grab & shuffle the list of iamges paths; extract labels from paths
"""
print("[INFO] loading images...") 
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# extract class labels & encode into numbers
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)
print("[INFO] encode %d categories in total" % len(set(labels)))


"""
- load ResNet50 without top
- noting that the "top" dropped from resnet includes the GlobalAveragePooling2D layer
- perform model surgery to add the GlobalAveragePooling2D
"""
if gloAvgPool == True:
    print("[INFO] loading ResNet50 & add GlobalAveragePooling2D...")
    x = Input(shape=(224, 224, 3))
    baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=x) 
    # add layer
    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel) 
    # merge model
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    # model architecture path
    arch_path = "./output/resnet50_architecture_with_GloAvgPool.png" 

    # output dim
    accDim = 2048
else:
    print("[INFO] loading ResNet50 without GlobalAveragePooling2D...")
    model = ResNet50(weights="imagenet", include_top=False)
    arch_path = "./output/resnet50_architecture_without_GloAvgPool.png" 

    # output dim
    accDim = 7 * 7 * 2048
print("[INFO] noting that, accDim = ", accDim) 

print("[INFO] plot model architecture...")
plot_model(model, to_file=arch_path, show_shapes=True)


"""
- initiate a HDF5 dataset writer
"""
# store the class label names in the dataset
# (batch, 2048) = output volume shape of last GlobalAveragePooling2D layer
dataset = HDF5DatasetWriter(args["output"], 
        (len(imagePaths), accDim),
        dataKey="features", 
        bufSize=args["buffer_size"],
        )
dataset.storeClassLabels(le.classes_)


"""
- loop over imagePaths to load in images in batch;
"""
# create a progress bar
widgets = ["Extracting Features :", progressbar.Percentage(), " ", \
        progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over imagePaths
for i in np.arange(0, len(imagePaths), bs):
    # get imagePaths & labels of current batch
    batchPaths = imagePaths[i : i + bs]
    batchLabels = labels[i : i + bs]
    batchImages = []

    # load images of current batch
    for j, imgPath in enumerate(batchPaths):
        # ResNet is trained on 224 * 224 images
        image = load_img(imgPath, target_size=inputShapeBanks[args["model"]])
        image = img_to_array(image)

        # add batch dim
        image = np.expand_dims(image, axis=0)
        # Special for ImageNet dataset => substracting mean RGB pixel intensity
        image = imagenet_utils.preprocess_input(image)
        
        batchImages.append(image)
        pass

    # stack vertically the images in the batch
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs) # scores before softmax

    # flatten features of each image w.r.t to output volume shape
    features = features.reshape((features.shape[0], accDim))
        
    # add the feature & labels into HDF5 dataset
    dataset.add(features, batchLabels)

    # update pbar
    pbar.update(i)
    pass

# close the dataset when finished
dataset.close()
pbar.finish()


