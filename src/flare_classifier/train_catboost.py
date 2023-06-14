import os
import pickle
import sys
import csv
import json

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
val = os.path.join(input, "val.parquet")
features = os.path.join(input, "feature_names.csv")
train_data = pd.read_parquet(train)
val_data = pd.read_parquet(val)

features_file = open(features, "r")
features_names = list(csv.reader(features_file, delimiter=","))[0]

X_train, y_train = train_data[features_names], train_data["is_flare"]
X_val, y_val = val_data[features_names], val_data["is_flare"]

clf = CatBoostClassifier(
    iterations=iter, 
    depth=depth, 
    learning_rate=lr, 
    loss_function=loss_func, 
    random_seed=seed, 
    verbose=True, 
    eval_metric='Precision'
)

output = os.path.join("models", "catboost")
os.makedirs(output, exist_ok=True)
output = os.path.join(output, "catboost.pickle")

clf.fit(X_train,
        y_train,
        eval_set=(X_val, y_val))

train_curve = clf.get_evals_result()
curve_path = os.path.join("models", "catboost", "train_curve.json")

with open(curve_path, "w") as fd:
    json.dump(train_curve, fd)

with open(output, "wb") as fd:
    pickle.dump(clf, fd)
