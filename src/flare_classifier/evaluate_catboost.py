from metrics import *
from thr_optimize import *

import os
import pickle
import sys
import csv

import pandas as pd

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model data real_data\n")
    sys.exit(1)

model_file = sys.argv[1]
train_path = os.path.join(sys.argv[2], "train.parquet")
test_path = os.path.join(sys.argv[2], "test.parquet")
val_path = os.path.join(sys.argv[2], "val.parquet")
features = os.path.join(sys.argv[2], "feature_names.csv")

features_file = open(features, "r")
features_names = list(csv.reader(features_file, delimiter=","))[0]

with open(model_file, "rb") as fd:
    model = pickle.load(fd)

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)
val_data = pd.read_parquet(val_path)
real_flares = pd.read_csv(os.path.join(sys.argv[3]))

X_train, y_train = train_data[features_names], train_data["is_flare"]
X_test, y_test = test_data[features_names], test_data["is_flare"]
X_val, y_val = val_data[features_names], val_data["is_flare"]
X_real, y_real = real_flares[features_names], real_flares["is_flare"]

scaler_path = os.path.join(sys.argv[2], "scaler.pickle")

with open(scaler_path, "rb") as fd:
    scaler = pickle.load(fd)

X_real = pd.DataFrame(scaler.transform(X_real), columns=features_names)

thr = optimal_threshold(model.predict(X_val, prediction_type="Probability")[:, 1], y_val)
model.set_probability_threshold(thr)
print(f"optimal threshold: {thr}")

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
real_pred = model.predict(X_real, prediction_type="Probability")
real_pred_label = model.predict(X_real)

evaluate_metrics("catboost", y_train, train_pred, "train")
evaluate_metrics("catboost", y_test, test_pred, "test")
evaluate_metrics("catboost", y_real, real_pred_label, "real")
evaluate_real("catboost", y_real, real_pred[:, 1])
