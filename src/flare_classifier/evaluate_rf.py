from metrics import *

import math
import os
import pickle
import sys
import csv

import pandas as pd
from matplotlib import pyplot as plt


if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model data\n")
    sys.exit(1)

model_file = sys.argv[1]
train_path = os.path.join(sys.argv[2], "train.parquet")
test_path = os.path.join(sys.argv[2], "test.parquet")
features = os.path.join(sys.argv[2], "feature_names.csv")

features_file = open(features, "r")
features_names = list(csv.reader(features_file, delimiter=","))[0]

with open(model_file, "rb") as fd:
    model = pickle.load(fd)

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

X_train, y_train = train_data[features_names], train_data["is_flare"]
X_test, y_test = test_data[features_names], test_data["is_flare"]

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

evaluate_metrics("random_forest", y_train, train_pred, "train")
evaluate_metrics("random_forest", y_test, test_pred, "test")
