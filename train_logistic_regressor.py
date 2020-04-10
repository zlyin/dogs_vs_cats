#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import h5py
import pickle   # serialization tool


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", required=True, help="version number of test")
parser.add_argument("-j", "--jobs", type=int, default=2, \
        help="# of jobs to run when tuning hyperparameters")
args = vars(parser.parse_args())


"""
- Open a HDF5 dataset and split the training & test dataset by index;
- Noting to shuffle samples BEFORE creating the dataset!
- Data before thres is training set; while the rest data is test set;
"""
VERSION = args["version"]
OUTPUT_FOLDER= os.path.sep.join(["./output/round2_fe_logreg", "test" + VERSION])
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
OUTPUT= os.path.sep.join([OUTPUT_FOLDER, "best_estimator.cpickle"])

JOBS = args["jobs"]
DB = h5py.File("./output/round2_fe_logreg/features_resnet50_gloAvgPool.hdf5", "r")
thres = int(DB["labels"].shape[0] * 0.80)


"""
- Fine tuning hyperparameter C for logistic regression;
- Initiate GridSearchCV as model, feed in a classifier object;
- Train the model on training set
"""
print("[INFO] building pipeline & tuning hyperparameters...")
logreg = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=5000)
pca = PCA()
pipe = Pipeline(steps=[("pca", pca), ("logreg", logreg)])

#params = {
#        # samller c = stronger regularization
#        "C" : [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1], 
#        }

params = {
        # params of pipelines can be set using "__" to seperate param names
        "pca__n_components" : [5, 15, 30, 60, 90, 120],
        # samller c = stronger regularization
        "logreg__C" : np.logspace(-2, -1, 20),
        }

model = GridSearchCV(
        #logreg, 
        pipe,
        params,
        scoring=["accuracy", "neg_log_loss"],
        cv=5,
        n_jobs=JOBS, 
        verbose=3,
        refit="neg_log_loss", # must the name of scorer
        )
model.fit(DB["features"][:thres], DB["labels"][:thres])
print("[INFO] best score is {}".format(model.best_score_))
print("[INFO] best hyperparameters are {}".format(model.best_params_))


## plot PCA spectrum
print("[INFO] building PCA spectrum...")
pca.fit(DB["features"][:thres])

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2)
ax0.set_ylabel("PCA explained variance ratio")
ax0.axvline(model.best_estimator_.named_steps["pca"].n_components, \
        linestyle=":", label="n_components chosen")

# for each n_components, find the best cls
results = pd.DataFrame(model.cv_results_)
print(results.columns)

components_col = "param_pca__n_components"
best_clfs = results.groupby(components_col).apply(lambda g : g.nlargest(1,
    "mean_test_neg_log_loss"))
best_clfs.plot(
        x=components_col, 
        y="mean_test_neg_log_loss",
        yerr="std_test_neg_log_loss",
        legend=False, 
        ax=ax1)
ax1.set_ylabel("Classification accuracy (val)")
ax1.set_xlabel("n_components")

plt.tight_layout()
plt.title("PCA Spectrum")
plt.savefig(os.path.sep.join([OUTPUT_FOLDER, "pca_spectrum.png"]))


## evaluate the model
print("[INFO] evaluating on val set...")
preds = model.predict(DB["features"][thres:])   # return predicted LABELS!
print(classification_report(preds, DB["labels"][thres:], \
        target_names=DB["label_names"]))

# compute the raw accuracy with extra precision
acc = accuracy_score(preds, DB["labels"][thres:])
print("[INFO] accuracy score : {}".format(acc))


## serialize the model to disk
if "cpickle" not in OUTPUT:
    raise ValueError("Output file format must be '.cpickle' format!")

print("[INFO] serializing best estimator to disk...")
with open(OUTPUT, "wb") as f:       # need to be "xxx.cpickle" file!
    f.write(pickle.dumps(model.best_estimator_))
f.close()

# close DB
DB.close()



