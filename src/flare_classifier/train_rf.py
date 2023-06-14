import os
import pickle
import sys
import csv
import yaml

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(open("params.yaml"))["train_rf"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train_rf.py data\n")
    sys.exit(1)

input = sys.argv[1]
seed = params["seed"]
n_estimators = params["n_estimators"]

train = os.path.join(input, "train.parquet")
features = os.path.join(input, "feature_names.csv")
train_data = pd.read_parquet(train)

features_file = open(features, "r")
features_names = list(csv.reader(features_file, delimiter=","))[0]

X_train, y_train = train_data[features_names], train_data["is_flare"]

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
clf.fit(X_train, y_train)

output = os.path.join("models", "random_forest")
os.makedirs(output, exist_ok=True)
output = os.path.join(output, "random_forest.pickle")

with open(output, "wb") as fd:
    pickle.dump(clf, fd)
