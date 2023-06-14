from metrics import *
from mlp_model import *
from thr_optimize import *

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


def eval_mlp(model, dataloader, probs=False, thr=0.5):
    test_pred = []
    test_true = []

    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch

            y_test_pred = model(data)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = (y_test_pred > thr).float()

            if probs:
                test_pred.extend(y_test_pred.numpy().squeeze())
            else:
                test_pred.extend(y_pred_tag.numpy().squeeze())
                
            test_true.extend(labels.numpy())
            
    predict = np.array(test_pred)
    y_true = np.array(test_true)
    
    return y_true, predict


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
    model_dict = torch.load(fd)
    
model = BinaryClassification(len(features_names))
model.load_state_dict(model_dict)
model.eval()

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)
val_data = pd.read_parquet(val_path)
real_flares = pd.read_csv(os.path.join(sys.argv[3]))

scaler_path = os.path.join(sys.argv[2], "scaler.pickle")

with open(scaler_path, "rb") as fd:
    scaler = pickle.load(fd)

X_real, y_real = real_flares[features_names], real_flares["is_flare"]
X_real = pd.DataFrame(scaler.transform(X_real), columns=features_names)
real_flares = pd.concat([X_real, y_real], axis=1)

test_dataset = FlaresDataset(test_data, features_names)
train_dataset = FlaresDataset(train_data, features_names)
val_dataset = FlaresDataset(val_data, features_names)
real_flares_dataset = FlaresDataset(real_flares, features_names)

batch_size = 1024
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
real_flares_dataloader = DataLoader(real_flares_dataset, batch_size=100, shuffle=False)

y_val, val_probs = eval_mlp(model.cpu(), val_dataloader, probs=True)
opt_thr = optimal_threshold(val_probs, y_val) 
print(f'optimal threshold: {opt_thr}')

evaluate_metrics("mlp", *eval_mlp(model.cpu(), test_dataloader, thr=opt_thr), "test")
evaluate_metrics("mlp", *eval_mlp(model.cpu(), train_dataloader, thr=opt_thr), "train")
evaluate_metrics("mlp", *eval_mlp(model.cpu(), real_flares_dataloader, thr=opt_thr), "real")
evaluate_real("mlp", *eval_mlp(model.cpu(), real_flares_dataloader, probs=True))