from metrics import *
from mlp_model import *

import math
import os
import pickle
import sys
import csv
import torch
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def eval_mlp(model, dataloader):
    test_pred = []
    test_true = []

    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch

            y_test_pred = model(data)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)

            test_pred.append(y_pred_tag.numpy())
            test_true.append(labels.numpy())

    predict = np.array([a.squeeze().tolist() for a in test_pred][0])
    y_true = np.array([a.squeeze().tolist() for a in test_true][0])

    return y_true, predict


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
    model_dict = torch.load(fd)
    
model = BinaryClassification(len(features_names))
model.load_state_dict(model_dict)
model.eval()

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

test_dataset = FlaresDataset(test_data, features_names)
train_dataset = FlaresDataset(train_data, features_names)

batch_size = 1024
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

evaluate_metrics("mlp", *eval_mlp(model.cpu(), test_dataloader), "test")
evaluate_metrics("mlp", *eval_mlp(model.cpu(), train_dataloader), "train")
