import os
import pickle
import sys
import csv

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier


params = yaml.safe_load(open("params.yaml"))["train_catboost"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train_rf.py data\n")
    sys.exit(1)

input = sys.argv[1]
seed = params["seed"]
iter = params["iterations"]
depth = params["depth"]
lr = params["learning_rate"]
loss_func = params["loss_function"]

train = os.path.join(input, "train.parquet")
features = os.path.join(input, "feature_names.csv")
train_data = pd.read_parquet(train)

features_file = open(features, "r")
features_names = list(csv.reader(features_file, delimiter=","))[0]

X_train, y_train = train_data[features_names], train_data["is_flare"]

clf = CatBoostClassifier(
    iterations=iter, depth=depth, learning_rate=lr, loss_function=loss_func, random_seed=seed, verbose=True
)

clf.fit(X_train, y_train)

output = os.path.join("models", "catboost")
os.makedirs(output, exist_ok=True)
output = os.path.join(output, "catboost.pickle")

with open(output, "wb") as fd:
    pickle.dump(clf, fd)
