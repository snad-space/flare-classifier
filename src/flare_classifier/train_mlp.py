import os
import pickle
import sys
import csv

from mlp_model import FlaresDataset, BinaryClassification

import numpy as np
import pandas as pd
import yaml
import torch

from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


@torch.inference_mode()
def get_correct_count(pred, labels):
    predicted = torch.round(torch.sigmoid(pred))
    return (predicted == labels).sum().item()


@torch.inference_mode()
def calculate_accuracy(model, data_loader, criterion):
    correct, total = 0, 0
    loss = 0
    for batch in data_loader:
        data, labels = batch
        data = data.to(device)
        pred = model(data).squeeze(1).cpu()
        loss += criterion(pred, labels)
        correct += get_correct_count(pred, labels)
        total += labels.size(0)

    return correct / total, loss.item() / len(data_loader)


def train(model, criterion, optimizer, num_epochs, total_step, train_dataloader, val_dataloader):
    for epoch in range(num_epochs):
        correct, total, ep_loss = 0, 0, 0

        for i, (data, labels) in enumerate(train_dataloader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(data).squeeze(1)
            loss = criterion(outputs, labels)
            correct += get_correct_count(outputs, labels)

            loss.backward()
            optimizer.step()

            total += labels.size(0)
            ep_loss += loss.item()

            if i % 100 == 0:
                accuracy, loss_val = calculate_accuracy(model, val_dataloader, criterion)
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Train Loss: {loss.item():.4f}")
                print(f"Val accuracy: {accuracy}, Val loss: {loss_val}")

    return 1


params = yaml.safe_load(open("params.yaml"))["train_mlp"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train_rf.py data\n")
    sys.exit(1)

input = sys.argv[1]
lr = params["lr"]
batch_size = params["batch_size"]
num_epochs = params["num_epochs"]

train_path = os.path.join(input, "train.parquet")
val_path = os.path.join(input, "val.parquet")

features = os.path.join(input, "feature_names.csv")
train_data = pd.read_parquet(train_path)
val_data = pd.read_parquet(val_path)

features_file = open(features, "r")
features_names = list(csv.reader(features_file, delimiter=","))[0]

train_dataset = FlaresDataset(train_data, features_names)
val_dataset = FlaresDataset(val_data, features_names)

n_input = len(features_names)
baseline_nn = BinaryClassification(n_input)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
total_step = len(train_dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = baseline_nn.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.train()
train(
    model,
    criterion,
    optimizer,
    num_epochs=num_epochs,
    total_step=total_step,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
)

output = os.path.join("models", "mlp")
os.makedirs(output, exist_ok=True)

output = os.path.join(output, "mlp.pickle")

with open(output, "wb") as fd:
    torch.save(model, fd)
