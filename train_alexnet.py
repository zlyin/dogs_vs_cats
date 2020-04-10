#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tfconfig)

import sys
sys.path.append("../Orca/nn/conv")
from alexnet import AlexNet
from alexnet2 import AlexNet2
sys.path.append("../Orca/preprocessing")
from simplepreprocessor import SimplePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from meanpreprocessor import MeanPreprocessor
from patchpreprocessor import PatchPreprocessor
from croppreprocessor import CropPreprocessor
sys.path.append("../Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator
sys.path.append("../Orca/callbacks")
from trainingmonitor import TrainingMonitor

import config.dogs_vs_cats_config as config

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import json
import matplotlib
matplotlib.use("Agg")


# parameters
BATCH_SIZE = 128

## initiate all image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# load in RGB mean values of training set
trainmeans = json.loads(open("./output/dogs_vs_cats_train_mean.json").read())
mp = MeanPreprocessor(trainmeans["R"], trainmeans["G"], trainmeans["B"])


## initiate HDF5DataGenerator for trainset, trainvalset
# initiate data augmentor for trainingset
aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        )

trainGen = HDF5DatasetGenerator(
        config.TRAIN_HDF5, 
        BATCH_SIZE,
        # extract Patch => want to learn discrimitive patterns
        # substract mean, convert to keras array
        preprocessors=[pp, mp, iap],
        aug=aug,
        classes=2,
        )

# initiate data augmentor for trainval set
valGen = HDF5DatasetGenerator(
        config.VAL_HDF5, 
        BATCH_SIZE,
        # RESIZE the org image => validate/test on the whole image
        # substract mean, convert to keras array
        preprocessors=[pp, mp, iap],
        classes=2,
        )


## initalize an optimizer & compile model
print("[INFO] compiling model...")
adam = Adam(lr=0.001)

#model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.002)
model = AlexNet2.build(width=227, height=227, depth=3, classes=2, reg=0.002)

model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

# build callbacks
figpath = os.path.sep.join([config.OUTPUT_PATH,
    "{}_learning_curve.png".format(os.getpid())])
tm = TrainingMonitor(figpath)

cptpath = os.path.sep.join([config.OUTPUT_PATH,
    "model-alexnet2-{epoch:03d}-{val_loss:0.4f}.hdf5"])
checkpt = ModelCheckpoint(cptpath, monitor="val_loss", mode="min", \
        save_best_only=True, verbose=1)

# train the NN
model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // BATCH_SIZE,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // BATCH_SIZE,
        epochs=75,
        max_queue_size=BATCH_SIZE,
        callbacks=[tm, checkpt],
        verbose=1)

# save the model to file
#print("[INFO] seralizing the model...")
#model.save(config.MODEL_PATH, overwrite=True)   # save the best

# close HDFDatasetGenerator
trainGen.close()
valGen.close()

print("[INFO] Done!")





