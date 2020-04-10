#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.compat.v1.Session(config=config)

import sys
sys.path.append("../Orca/preprocessing")
from aspectawarepreprocessor import AspectAwarePreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from patchpreprocessor import PatchPreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
sys.path.append("../Orca/callbacks")
from trainingmonitor import TrainingMonitor
sys.path.append("../Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.applications import VGG16, ResNet50
from keras.models import Model  # important!
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, \
        BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from imutils import paths
#from config import train_resnet50_config as cfg
import argparse
import json
import numpy as np


## cache variables
NUM_CLASSES = 2
TRAIN_HDF5 = "./data/train.hdf5"
TRAINVAL_HDF5 = "./data/trainval.hdf5"
VAL_HDF5 = "./data/val.hdf5"
TEST_HDF5 = "./data/test.hdf5"
DATASET_MEAN = "./output/dogs_vs_cats_mean.json"

#MODEL_PATH = "./output/round3_resnet50.model"
OUTPUT_PATH = "./output/test6_aspect_meansub_imgtoarr_version2aug"

BATCH_SIZE = 128
EPOCHS = 5


## initiate image preprocessors
sp = SimplePreprocessor(224, 224)
pp = PatchPreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
aap = AspectAwarePreprocessor(224, 224)

# load in RGB mean values of training set
trainmeans = json.loads(open("./output/dogs_vs_cats_mean.json").read())
mp = MeanPreprocessor(trainmeans["R"], trainmeans["G"], trainmeans["B"])

# initiate HDF5DataGenerator for trainset, trainvalset

# version 1
#aug = ImageDataGenerator(
#        rotation_range=20,
#        zoom_range=0.15,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        shear_range=0.15,
#        horizontal_flip=True,
#        fill_mode="nearest",
#        )
#

# version 2
aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        )

trainGen = HDF5DatasetGenerator(
        TRAIN_HDF5, 
        BATCH_SIZE,
        # extract Patch => want to learn discrimitive patterns
        # substract mean, convert to keras array
        #preprocessors=[pp, mp, iap],
        #preprocessors=[pp, iap],
        #preprocessors=[sp, iap],
        #preprocessors=[sp, mp, iap],
        preprocessors=[aap, mp, iap],
        aug=aug,
        classes=2,
        )

# initiate data augmentor for trainval set
trainvalGen = HDF5DatasetGenerator(
        TRAINVAL_HDF5, 
        BATCH_SIZE,
        # RESIZE the org image => validate/test on the whole image
        # substract mean, convert to keras array
        #preprocessors=[pp, mp, iap],
        #preprocessors=[pp, iap],
        #preprocessors=[sp, iap],
        preprocessors=[aap, mp, iap],
        classes=2,
        )


## perform model surgery, load ResNet50 network without head layers, explicitly define input_tensor
x = Input(shape=(224, 224, 3))
backbone = ResNet50(weights="imagenet", include_top=False, input_tensor=x) 
# FREEZE the backbone & train the new head layers for 10-30 epochs
for layer in backbone.layers:
    layer.trainable = False

head = backbone.output
head = BatchNormalization(axis=-1)(head)
head = GlobalAveragePooling2D()(head)
head = Dense(NUM_CLASSES, activation="softmax")(head)
model = Model(inputs=backbone.input, outputs=head)

print("[INFO] plot model architecture...")
arch_path = os.path.join(OUTPUT_PATH, "resnet50_new_head.png")
plot_model(model, to_file=arch_path, show_shapes=True)


## initalize an optimizer & compile model & train NN
print("[INFO] compiling model for new head layers warm-up...")
#opt = RMSprop(lr=0.001)
opt = Adam(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# build callbacks
figpath = os.path.sep.join([OUTPUT_PATH,
    "{}_learning_curve.png".format(os.getpid())])
tm = TrainingMonitor(figpath)

cptpath = os.path.sep.join([OUTPUT_PATH,
    "model-resnet50_new_head-{epoch:03d}-{val_loss:0.4f}-" + str(os.getpid()) + ".hdf5"])
ckpt = ModelCheckpoint(cptpath, monitor="val_loss", mode="min", \
        save_best_only=True, verbose=1)

# train 
model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // BATCH_SIZE + 1,
        validation_data=trainvalGen.generator(),
        validation_steps=trainvalGen.numImages // BATCH_SIZE + 1, 
        epochs=EPOCHS,
        max_queue_size=BATCH_SIZE,
        callbacks=[tm, ckpt],
        verbose=1)


# close HDFDatasetGenerator
trainGen.close()
trainvalGen.close()

print("[INFO] Done!")




"""just in case need to unfreeze deeper layers!"""

## UNfreeze the CONV layers NEAR the new head & train to localize the weights
#print("[INFO] fine tune the model in %d stages" % len(args["fine_tune_layers"]))
#for i, num in enumerate(args["fine_tune_layers"]):
#    # unfree layers
#    index = int(num)
#    for layer in model.layers[index:]:
#        layer.trainable = True
#
#    print("\n[INFO] stage %d/%d, unfreez layers from %d & recompile model..." %
#            (i + 1, len(args["fine_tune_layers"]), index))
#    sgd = SGD(lr=0.001)
#    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
#
#    print("[INFO] fine tuning the model...")
#    model.fit_generator(aug.flow(trainX, trainY, batch_size=32), \
#            validation_data=(testX, testY), \
#            steps_per_epoch=len(trainX) // 32, \
#            epochs=50, verbose=1)
#   
#    print("[INFO] evaluating...")
#    predictions = model.predict(testX, batch_size=32)
#    print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), \
#            target_names=classNames))
#    
#    # save current model
#    outputName = "fine_tune_stage_%d_from_layer_%d.hdf5" % (i + 1, index)
#    filepath = os.path.join(args["output_dir"], outputName)
#    model.save(filepath)
#    pass
#
#
#
