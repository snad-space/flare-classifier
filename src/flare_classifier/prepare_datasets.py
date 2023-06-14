import os
import pickle
import sys
import csv

import yaml
import numpy as np
import pandas as pd

from data_preprocessing import generate_datasets
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open("params.yaml"))["prepare_datasets"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 prepare_datasets.py positive_class negative_class\n")
    sys.exit(1)

train_size = params["train_size"]
test_size = params["test_size"]
rng = np.random.default_rng(params["seed"])

positive_class = sys.argv[1]
negative_class = sys.argv[2]

folder = os.path.join("data", "prepared")
output_train = os.path.join(folder, "train.parquet")
output_test = os.path.join(folder, "test.parquet")
output_val = os.path.join(folder, "val.parquet")
output_features = os.path.join(folder, "feature_names.csv")
output_scaler = os.path.join(folder, "scaler.pickle")

train_features, test_features, val_features, feature_names = generate_datasets(
    positive_class, negative_class, train_size=train_size, test_size=test_size, random_seed=rng
)

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

scaler = StandardScaler()
train_features[feature_names] = scaler.fit_transform(train_features[feature_names])
test_features[feature_names] = scaler.transform(test_features[feature_names])
val_features[feature_names] = scaler.transform(val_features[feature_names])

train_features.to_parquet(output_train)
test_features.to_parquet(output_test)
val_features.to_parquet(output_val)

with open(output_features, "w") as f:
    write = csv.writer(f)

    write.writerow(feature_names)

with open(output_scaler, "wb") as f:
    pickle.dump(scaler, f)
